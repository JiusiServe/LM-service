# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import random

from typing import Optional
from llm_service.kv_cache_tracer import KVCacheSupervisor


class RoutingContext:
    def __init__(
        self,
        endpoints: list[int],
        request_stats: dict,
        prefix_info: Optional[KVCacheSupervisor] = None,
        token_ids: Optional[list[int]] = None,
        mm_hashes: Optional[list[str]] = None,
    ):
        self.endpoints = endpoints
        self.request_stats = request_stats
        self.prefix_info = prefix_info
        self.token_ids = token_ids
        self.mm_hashes = mm_hashes


class RoutingInterface:
    def route_request(self, rctx: RoutingContext) -> int:
        """
        Route the request to a specific instance based on the request stats.
        It can also be based on engine stats in the future.

        Args:
            endpoints (list[int]): The list of instance IDs.
            request_stats (dict): The incoming request stats.

        Returns:
            int: The ID of the selected instance.
        """

        # Implement your routing logic here
        raise NotImplementedError("Subclasses should implement this method.")


class RandomRouter(RoutingInterface):
    def route_request(self, rctx: RoutingContext) -> int:
        if not rctx.endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        return random.choice(rctx.endpoints)


class RoundRobinRouter(RoutingInterface):
    def __init__(self):
        self.current_index = 0

    def route_request(self, rctx: RoutingContext) -> int:
        if not rctx.endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        selected_index = self.current_index % len(rctx.endpoints)
        self.current_index = selected_index + 1
        return rctx.endpoints[selected_index]


class LeastInFlightRouter(RoutingInterface):
    def route_request(self, rctx: RoutingContext) -> int:
        if not rctx.endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")

        def get_in_flight_count(endpoint_id: int) -> int:
            stats = rctx.request_stats.get(endpoint_id)
            return len(stats.in_flight_requests) if stats else 0

        return min(rctx.endpoints, key=get_in_flight_count)


class PrefixAwareRouter(RandomRouter):
    def route_request(self, rctx: RoutingContext) -> int:
        if not rctx.endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        assert rctx.prefix_info is not None
        assert rctx.token_ids is not None
        assert rctx.mm_hashes is not None

        max_prefix_blocks = 0
        candidate = []
        for iid in rctx.endpoints:
            prefix_blocks = rctx.prefix_info.find_prefix_block(
                iid, rctx.token_ids, rctx.mm_hashes
            )
            if prefix_blocks == max_prefix_blocks:
                candidate.append(iid)
            elif prefix_blocks > max_prefix_blocks:
                candidate = [iid]
                max_prefix_blocks = prefix_blocks

        return super().route_request(
            RoutingContext(
                endpoints=candidate, request_stats=rctx.request_stats
            )
        )
