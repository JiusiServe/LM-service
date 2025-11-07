# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
import msgspec
import zmq
from vllm.distributed.kv_events import BlockStored, BlockRemoved, KVEventBatch
import asyncio
from typing import Optional


class KVBlock:
    def __init__(
        self,
        token_ids: list[int],
        mm_hashes: list[str],
        block_size: int,
        block_hash: str,
        child_block_hash: Optional[str] = None,
        parent_block_hash: Optional[str] = None,
        token_hash: Optional[int] = None,
    ):
        self.token_ids = token_ids
        self.mm_hashes = mm_hashes
        self.block_size = block_size

        self.block_hash = block_hash
        self.child_block_hash = child_block_hash
        self.parent_block_hash = parent_block_hash

        self.token_hash = token_hash


class KVCacheTracer:
    def __init__(self, instance_id, zmq_endpoint):
        self.instance_id = instance_id
        self.zmq_endpoint = zmq_endpoint
        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

        self.token_to_kvblock: dict[int, list[KVBlock]] = {}
        self.hash_to_kvblock: dict[str, KVBlock] = {}
        self.block_size = None

    async def subscribe_to_instance(self):
        context = zmq.asyncio.Context()
        sub = context.socket(zmq.SUB)
        sub.connect(self.zmq_endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, "kv-events")

        while True:
            _, seq_bytes, payload = await sub.recv_multipart()
            event_batch = self.decoder.decode(payload)
            self.process_events(event_batch)

    def process_events(self, event_batch):
        for event in event_batch.events:
            if isinstance(event, BlockStored):
                self._add_block(event)
            elif isinstance(event, BlockRemoved):
                self._remove_block(event)

    def _add_block(self, block: BlockRemoved):
        for idx, bh in enumerate(block.block_hashes):
            if self.block_size is None:
                self.block_size = block.block_size
            self.hash_to_kvblock[bh] = KVBlock(
                token_ids=block.token_ids[
                    idx * block.block_size : (idx + 1) * block.block_size
                ],
                mm_hashes=block.mm_hashes[idx],
                block_size=block.block_size,
                block_hash=bh,
                child_block_hash=None
                if idx == len(block.block_hashes) - 1
                else block.block_hashes[idx + 1],
                parent_block_hash=block.parent_block_hash
                if idx == 0
                else block.block_hashes[idx - 1],
            )

        if block.parent_block_hash is not None:
            self.hash_to_kvblock[
                block.parent_block_hash
            ].child_block_hash = block.block_hashes[0]
        else:
            first_block = self.hash_to_kvblock[block.block_hashes[0]]
            token_hash = self._token_hash(first_block.token_ids)
            first_block.token_hash = token_hash
            self.token_to_kvblock[token_hash] = self.token_to_kvblock.get(
                first_block.token_hash, []
            ) + [first_block]

    def _remove_block(self, block: BlockRemoved):
        for bh in block.block_hashes:
            kvblock = self.hash_to_kvblock.pop(bh, None)
            if not kvblock:
                continue

            if kvblock.parent_block_hash is None:
                assert kvblock.token_hash is not None
                kvblock_list = self.token_to_kvblock.pop(kvblock.token_hash, [])
                kvblock_list = [
                    b
                    for b in kvblock_list
                    if b.block_hash != kvblock.block_hash
                ]
                if kvblock_list:
                    self.token_to_kvblock[kvblock.token_hash] = kvblock_list
            else:
                self.hash_to_kvblock[
                    kvblock.parent_block_hash
                ].child_block_hash = None

            while kvblock.child_block_hash is not None:
                kvblock = self.hash_to_kvblock.pop(kvblock.child_block_hash)

    def find_prefix_block(self, tokens: list[int], mm_hashes: list[str]) -> int:
        if self.block_size is None:
            return 0
        max_block_num = len(tokens) // self.block_size
        if max_block_num == 0:
            return 0
        th = self._token_hash(tokens[0 : self.block_size])
        candidates = self.token_to_kvblock.get(th, [])
        first_kvblock = next(
            (
                x
                for x in candidates
                if self._matching_condition(
                    x, tokens[0 : self.block_size], mm_hashes, 0
                )
            ),
            None,
        )
        if first_kvblock is None:
            return 0

        matched_mm_hashes = len(first_kvblock.mm_hashes)
        matched_kvblocks = 1
        cur_kvblock = first_kvblock

        while cur_kvblock.child_block_hash is not None:
            cur_kvblock = self.hash_to_kvblock[cur_kvblock.child_block_hash]
            if not self._matching_condition(
                cur_kvblock,
                tokens[
                    matched_kvblocks * self.block_size : (matched_kvblocks + 1)
                    * self.block_size
                ],
                mm_hashes,
                matched_mm_hashes,
            ):
                return matched_kvblocks

            matched_mm_hashes += len(cur_kvblock.mm_hashes)
            matched_kvblocks += 1

        return matched_kvblocks

    def _matching_condition(
        self,
        kv_block: KVBlock,
        tokens: list[int],
        mm_hashes: list[str],
        matched_mm_hashes: int,
    ):
        if kv_block.token_ids != tokens:
            return False
        for mm_hash in kv_block.mm_hashes:
            if mm_hash != mm_hashes[matched_mm_hashes]:
                return False
            matched_mm_hashes += 1
        return True

    def _token_hash(self, tokens: list[int]) -> int:
        return hash(tuple(tokens))


class KVCacheSupervisor:
    def __init__(self):
        self.tracers: dict[int, KVCacheTracer] = {}

    def add_instance(self, instance_id: int, zmq_endpoint: str):
        tracer = KVCacheTracer(instance_id, zmq_endpoint)
        self.tracers[instance_id] = tracer

    def start_all(self):
        for t in self.tracers.values():
            asyncio.create_task(t.subscribe_to_instance())

    def find_prefix_block(
        self, instance_id: int, tokens: list[int], mm_hashes: list[str]
    ) -> int:
        tracer = self.tracers.get(instance_id)
        if not tracer:
            raise ValueError(f"Instance {instance_id} not found")
        return tracer.find_prefix_block(tokens, mm_hashes)
