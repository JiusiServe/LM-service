# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import signal
import socket
import uvicorn
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any
import uvloop
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from lm_service.entrypoints.cli_args import make_arg_parser
from lm_service.apis.vllm.proxy import Proxy
from lm_service.logger_utils import init_logger
import lm_service.envs as lm_service_envs
from vllm.entrypoints.openai.api_server import (
    validate_json_request,
    ChatCompletionRequest,
    setup_server,
    build_app,
    init_app_state,
    chat,
    base,
    engine_client,
)
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    ChatCompletionResponse,
)
from vllm.entrypoints.utils import with_cancellation
from vllm.engine.protocol import EngineClient
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser, decorate_logs
from vllm.usage.usage_lib import UsageContext


logger = init_logger(__name__)

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_chat_completion(
    request: ChatCompletionRequest, raw_request: Request
):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    if lm_service_envs.TIMECOUNT_ENABLED:
        # wait for logging
        proxy_client = engine_client(raw_request)
        asyncio.create_task(proxy_client.log_metrics())
    # non-streaming response
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())
    # streaming response
    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.get("/v1/metrics")
@with_cancellation
async def metrics(raw_request: Request):
    proxy_client = engine_client(raw_request)
    try:
        asyncio.create_task(proxy_client.log_metrics())
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    decorate_logs("APIServer")
    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    **uvicorn_kwargs: Any,
):
    """Serve the HTTP app using Uvicorn."""
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()

    # start the server
    server_task = loop.create_task(
        server.serve(sockets=[sock] if sock else None)
    )

    # handle SIGTERM for graceful shutdown
    def signal_handler() -> None:
        server_task.cancel()

    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async def dummy_shutdown() -> None:
        pass

    try:
        await server_task
        return dummy_shutdown
    except asyncio.CancelledError:
        logger.info("HTTP server has been cancelled.")
        return server.shutdown()


async def run_server_worker(
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    **uvicorn_kwargs,
) -> None:
    """Run a single worker of the API server."""
    async with build_async_proxy_client(args) as proxy_client:
        app = build_app(args)
        vllm_config = proxy_client.vllm_config
        await init_app_state(proxy_client, vllm_config, app.state, args)
        await proxy_client.start_output_handler_once()
        app.include_router(router)  # add API routes in this file
        logger.info(
            "Starting vLLM API server %d on %s",
            proxy_client.vllm_config.parallel_config._api_process_rank,
            listen_address,
        )
        shutdown_task = await serve_http(app, sock, **uvicorn_kwargs)
    try:
        await shutdown_task()
    finally:
        sock.close()


@asynccontextmanager
async def build_async_proxy_client(
    args: Namespace,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
) -> AsyncIterator[EngineClient]:
    """Build the async engine client."""
    proxy_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = proxy_args.create_engine_config(usage_context=usage_context)
    p = Proxy(
        vllm_config=vllm_config,
        proxy_addr=args.proxy_addr,
        encode_addr_list=args.encode_addr_list,
        pd_addr_list=args.pd_addr_list,
        p_addr_list=args.p_addr_list,
        d_addr_list=args.d_addr_list,
        transfer_protocol=args.transfer_protocol,
        metastore_client_config=args.metastore_client_config,
        model_name=args.model,
        enable_health_monitor=False,
    )
    yield p
    p.shutdown()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
