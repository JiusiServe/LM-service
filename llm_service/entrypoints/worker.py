# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import uvloop
from llm_service.stats_loggers import DisaggWorkerStatsLogger
from llm_service.workers.vllm.disagg_worker import DisaggWorker
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION
import llm_service.envs as llm_service_envs
import signal
from llm_service.logger_utils import init_logger

logger = init_logger(__name__)


async def run(args, engine: EngineClient):
    logger.info("Initializing disaggregated worker")

    def signal_handler(*_):
        raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)

    worker = DisaggWorker(
        engine=engine,
        address=args.worker_addr,
        proxy_addr=args.proxy_addr,
    )

    try:
        await worker.run_busy_loop()
    finally:
        worker.shutdown()


async def main(args) -> None:
    logger.info("Disaggregated Worker Server, vLLM ver. %s", VLLM_VERSION)
    logger.info("Args: %s", args)

    stat_loggers = None
    if llm_service_envs.TIMECOUNT_ENABLED:
        stat_loggers = [DisaggWorkerStatsLogger]
        logger.info("Time counting is enabled.")

    engine_args = AsyncEngineArgs.from_cli_args(args)

    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=stat_loggers)
    try:
        await run(args, engine)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--proxy-addr",
        type=str,
        required=True,
        help="The address of the proxy.",
    )
    parser.add_argument(
        "--worker-addr",
        type=str,
        required=True,
        help="The address of the worker.",
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="Disable MQLLMEngine for AsyncLLMEngine.",
    )
    AsyncEngineArgs.add_cli_args(parser)
    uvloop.run(main(parser.parse_args()))
