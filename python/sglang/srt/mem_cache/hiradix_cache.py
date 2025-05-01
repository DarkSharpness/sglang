from collections import defaultdict
import heapq
import logging
import threading
import time
from typing import Dict, List, Optional

import torch
import hiradix_schedule_utils

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolHost,
    MLATokenToKVPool,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode, SortNode, EVICT_THRESHOLD, MAX_EVICT

logger = logging.getLogger(__name__)


class HiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        evict_policy: str,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        hicache_oracle: bool = False,
    ):
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.max_size: int = self.kv_cache.size # type: ignore
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                page_size,
                pin_memory=False if hicache_oracle else True,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                page_size,
                pin_memory=False if hicache_oracle else True,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = tp_cache_group

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            page_size,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
            oracle=hicache_oracle,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.tree_cpp = hiradix_schedule_utils.HiRadixCache_CPP()
        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
            evict_policy=evict_policy,
            disable=False
        )
        if evict_policy == "fair":
            assert self.cache_controller.write_policy != "write_back", (
                "Don't support fair eviction with write back policy yet"
            )

    def reset(self):
        TreeNode.reset_counter()
        self.root_node = TreeNode()
        self.tree_cpp.reset()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.load_cache_event.clear()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def write_backup(self, node: TreeNode):
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.tree_cpp.backup_node(node.id)
            self.ongoing_write_through[node.id] = node
            self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def inc_hit_count(self, node: TreeNode):
        if node.backuped or self.cache_controller.write_policy == "write_back":
            return
        node.hit_count += 1
        if node.hit_count >= self.write_through_threshold:
            self.write_backup(node)
            node.hit_count = 0

    def inc_lock_ref_cpp(self, node: TreeNode):
        self.tree_cpp.inc_lock_ref(node.id)

    def dec_lock_ref_cpp(self, node: TreeNode):
        self.tree_cpp.dec_lock_ref(node.id)

    def insert_pending_request(self, node: TreeNode, key: List[int]):
        self.tree_cpp.insert_pending_request(node.id, key)

    def writing_check(self):
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(queue_size.item()):
            ack_id = self.cache_controller.ack_write_queue.get()
            self.dec_lock_ref(self.ongoing_write_through[ack_id])
            del self.ongoing_write_through[ack_id]

    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
                start_node, end_node = self.ongoing_load_back[ack_id]
                self.dec_lock_ref(end_node)
                while end_node != start_node:
                    assert end_node.loading
                    end_node.loading = False
                    end_node = end_node.parent
                # clear the reference
                del self.ongoing_load_back[ack_id]
            except Exception:
                break

    def evictable_size(self):
        return self.evictable_size_

    def _evict_fair_hicache(self, num_tokens: int):
        tree_list, rank_map = self._collect_leaves_and_rank_device()
        num_evicted = 0
        leaves = [SortNode(node, rank_map) for node in tree_list]
        heapq.heapify(leaves)
        evicted: Dict[TreeNode, int] = defaultdict(int)
        while num_evicted < num_tokens and len(leaves):
            y = heapq.heappop(leaves)
            x = y.node
            if x.lock_ref > 0:
                continue
            assert x.value is not None, "can't evict a node without value"
            old_evict = evicted[x]
            if (num_remain := len(x.key)) - old_evict <= EVICT_THRESHOLD or x.host_value is None:
                # evict the whole node
                num_evicted += num_remain - old_evict
                evicted.pop(x, None)
                if x.host_value is None: # no backup
                    self.token_to_kv_pool_allocator.free(x.value)
                    self._delete_leaf(x)
                    self.tree_cpp.delete_node(x.id)
                else:
                    assert self.cache_controller.evict_device(x.value, x.host_value) == num_remain
                    self.evictable_size_ -= num_remain
                    x.value = None
                    self.tree_cpp.evict_node(x.id)

                for child in x.parent.children.values():
                    if not child.evicted:
                        break
                else:
                    x.parent._evict_count = x._evict_count + num_remain
                    heapq.heappush(leaves, SortNode(x.parent, rank_map))
            else:
                num_evicted += MAX_EVICT
                evicted[x] += MAX_EVICT
                y.evict_max()
                heapq.heappush(leaves, y)

        for x, num_delete in evicted.items():
            num_remain = len(x.key) - num_delete
            assert x.value is not None and x.host_value is not None

            # need to split the node, the remaining part is not evictable
            z = TreeNode()
            z.lock_ref = x.lock_ref
            z.loading = x.loading
            z.key, x.key = x.key[:num_remain], x.key[num_remain:]
            z.value, x.value = x.value.split((num_remain, num_delete))
            z.host_value, x.host_value = x.host_value.split((num_remain, num_delete))
            z.children = {self.get_child_key_fn(x.key): x}
            z.parent = x.parent
            z.parent.children[self.get_child_key_fn(z.key)] = z
            x.parent = z
            z.last_access_time = x.last_access_time
            z._evict_count = x._evict_count
            assert x.value is not None and x.host_value is not None
            assert self.cache_controller.evict_device(x.value, x.host_value) == num_delete
            self.evictable_size_ -= num_delete
            x.value = None
            self.tree_cpp.split_node(x.id, x.key, z.id, z.key)
            self.tree_cpp.evict_node(x.id)

    def evict(self, num_tokens: int):
        if self.evict_policy == "fair" and self.evictable_size_ > self.max_size // 4:
            return self._evict_fair_hicache(num_tokens)

        leaves = self._collect_leaves_device()
        heapq.heapify(leaves)

        num_evicted = 0
        pending_nodes = []
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    num_evicted += self.write_backup(x)
                    pending_nodes.append(x)
                else:
                    num_evicted += self._evict_write_through_selective(x)
            else:
                num_evicted += self._evict_write_through(x)

            for child in x.parent.children.values():
                if child in pending_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

        if self.cache_controller.write_policy == "write_back":
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                self.writing_check()
                time.sleep(0.1)

            for node in pending_nodes:
                assert node.backuped
                self._evict_write_through(node)

    def _evict_write_through(self, node: TreeNode):
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self.tree_cpp.evict_node(node.id)
        return num_evicted

    def _evict_write_through_selective(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        self.tree_cpp.delete_node(node.id)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            num_evicted += self.cache_controller.evict_host(x.host_value)

            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]
            self.tree_cpp.delete_node(x.id)
            x.dangling = True

            if len(x.parent.children) == 0 and x.parent.evicted:
                heapq.heappush(leaves, x.parent)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            self.tree_cpp.load_node(node.id)
            offset += len(node.host_value)
            node.loading = True
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def init_load_back(
        self,
        last_node: TreeNode,
        prefix_indices: torch.Tensor,
        mem_quota: Optional[int] = None,
    ):
        assert (
            len(prefix_indices) == 0 or prefix_indices.is_cuda
        ), "indices of device kV caches should be on GPU"
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                prefix_indices = (
                    loading_values
                    if len(prefix_indices) == 0
                    else torch.cat([prefix_indices, loading_values])
                )
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )

            while last_node.evicted:
                last_node = last_node.parent

        return last_node, prefix_indices

    def ready_to_load_cache(self):
        self.load_cache_event.set()

    def match_prefix(self, key: List[int], include_evicted=False, **kwargs):
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.disable or len(key) == 0:
            if include_evicted:
                return empty_value, self.root_node, self.root_node
            else:
                return empty_value, self.root_node

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        last_node_global = last_node
        while last_node.evicted:
            last_node = last_node.parent

        if include_evicted:
            return value, last_node, last_node_global
        else:
            return value, last_node

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.access()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.access()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.loading = child.loading
        new_node.hit_count = child.hit_count
        new_node._evict_count = child._evict_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        self.tree_cpp.split_node(child.id, child.key, new_node.id, new_node.key)
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.access()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.access()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len]
                    self.tree_cpp.load_node(node.id)
                    self.token_to_kv_pool_host.update_synced(node.host_value)
                    self.evictable_size_ += len(node.value)
                else:
                    self.inc_hit_count(node)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted:
                    new_node.value = value[:prefix_len]
                    self.tree_cpp.load_node(new_node.id)
                    self.token_to_kv_pool_host.update_synced(new_node.host_value)
                    self.evictable_size_ += len(new_node.value)
                else:
                    self.inc_hit_count(new_node)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self.tree_cpp.insert_node(new_node.id, new_node.key, node.id)

            if self.cache_controller.write_policy != "write_back":
                self.inc_hit_count(new_node)
        return total_prefix_length

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list

    def _collect_leaves_and_rank_device(self):
        def is_leaf(node: TreeNode):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list: List[TreeNode] = []
        time_list: List[float] = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            time_list.append(cur_node.last_access_time)
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                stack.extend(v for v in cur_node.children.values() if not v.evicted)

        time_list = sorted(time_list)
        return ret_list, {t: i for i, t in enumerate(time_list)}
