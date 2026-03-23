# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import asyncio
import os
import uuid
import numpy as np
from .conftest import load_config

try:
    import pytest
    from PIL import Image
except ImportError:
    pass


try:
    from vllm.multimodal.image import convert_image_mode
    from ..conftest import RemoteEPDServer
    from ..nightly.multi_node.config.utils import get_cluster_ips
    from ..nightly.multi_node.config.multi_node_epd_config import (
        ClusterManager,
        EnvManager,
    )
    from vllm.utils import get_open_port
    from vllm import SamplingParams
    from lm_service.apis.vllm.proxy import Proxy
    from pathlib import Path
except (ImportError, ModuleNotFoundError):
    pass


model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")
CONTAINER_NAME = load_config().get("container_name")

TENSOR_PARALLELS = [1]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"
ENABLE_PREFIX = [True, False]

DATASET_NAME = ["simulate_truth"]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "what is the brand of this camera?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.function
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_lm_service_request_timeout_seconds_001(model: str, tp_size: int):
    """
    LM_SERVICE_REQUEST_TIMEOUT_SECONDS is empty; generate fails and returns an error.
    """
    image = convert_image_mode(
        Image.open(
            Path(__file__).parent.parent.parent.parent / "tools" / "224.png"
        ),
        "RGB",
    )
    IMAGE_ARRAY = np.array(image)
    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"127.0.0.1:{get_open_port()}"
    e_addr = f"127.0.0.1:{get_open_port()}"
    pd_addr = f"127.0.0.1:{get_open_port()}"

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--worker-addr",
        f"{e_addr}",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "10000",
        "--max-num-batched-tokens",
        "10000",
        "--worker-addr",
        f"{pd_addr}",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        f"{mooncake_ip}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "10000",
        "--eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
        "--metrics_port",
        str(metrics_port),
    ]

    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        pd_num=1,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as server:
        print("vllm instance is ready")
        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=[e_addr],
            pd_addr_list=[pd_addr],
            model_name=model,
            enable_health_monitor=True,
        )
        print("proxy is success")
        try:
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(max_tokens=128, temperature=0.0),
                request_id=str(uuid.uuid4()),
            )
            async for o in outputs:
                print(f"{o.outputs}", flush=True)
            p.shutdown()
        except Exception:
            assert server.check_log("invalid literal", 120), "init success"


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.function
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_lm_service_request_timeout_seconds_002(model: str, tp_size: int):
    """
    LM_SERVICE_REQUEST_TIMEOUT_SECONDS is "aaa"; generate fails and returns an error.
    """
    image = convert_image_mode(
        Image.open(
            Path(__file__).parent.parent.parent.parent / "tools" / "224.png"
        ),
        "RGB",
    )
    IMAGE_ARRAY = np.array(image)
    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "aaa",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"127.0.0.1:{get_open_port()}"
    e_addr = f"127.0.0.1:{get_open_port()}"
    pd_addr = f"127.0.0.1:{get_open_port()}"

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--worker-addr",
        f"{e_addr}",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "10000",
        "--max-num-batched-tokens",
        "10000",
        "--worker-addr",
        f"{pd_addr}",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        f"{mooncake_ip}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "10000",
        "--eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
        "--metrics_port",
        str(metrics_port),
    ]

    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        pd_num=1,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as server:
        print("vllm instance is ready")
        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=[e_addr],
            pd_addr_list=[pd_addr],
            model_name=model,
            enable_health_monitor=True,
        )
        print("proxy is success")
        try:
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(max_tokens=128, temperature=0.0),
                request_id=str(uuid.uuid4()),
            )
            async for o in outputs:
                print(f"{o.outputs}", flush=True)
            p.shutdown()
        except Exception:
            assert server.check_log("invalid literal", 120), "init success"


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.function
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_lm_service_request_timeout_seconds_003(model: str, tp_size: int):
    """
    LM_SERVICE_REQUEST_TIMEOUT_SECONDS is "0"; generate fails and returns an error.
    """
    image = convert_image_mode(
        Image.open(
            Path(__file__).parent.parent.parent.parent / "tools" / "224.png"
        ),
        "RGB",
    )
    IMAGE_ARRAY = np.array(image)
    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "0",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"127.0.0.1:{get_open_port()}"
    e_addr = f"127.0.0.1:{get_open_port()}"
    pd_addr = f"127.0.0.1:{get_open_port()}"

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--worker-addr",
        f"{e_addr}",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "10000",
        "--max-num-batched-tokens",
        "10000",
        "--worker-addr",
        f"{pd_addr}",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        f"{mooncake_ip}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "10000",
        "--eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
        "--metrics_port",
        str(metrics_port),
    ]

    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        pd_num=1,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as server:
        print("vllm instance is ready")
        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=[e_addr],
            pd_addr_list=[pd_addr],
            model_name=model,
            enable_health_monitor=True,
        )
        print("proxy is success")
        try:
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(max_tokens=128, temperature=0.0),
                request_id=str(uuid.uuid4()),
            )
            async for o in outputs:
                print(f"{o.outputs}", flush=True)
            p.shutdown()
        except Exception:
            assert server.check_log("invalid literal", 120), "init success"


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.function
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_lm_service_request_timeout_seconds_004(model: str, tp_size: int):
    """
    LM_SERVICE_REQUEST_TIMEOUT_SECONDS is "-1"; generate fails and returns an error.
    """
    image = convert_image_mode(
        Image.open(
            Path(__file__).parent.parent.parent.parent / "tools" / "224.png"
        ),
        "RGB",
    )
    IMAGE_ARRAY = np.array(image)
    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "-1",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"127.0.0.1:{get_open_port()}"
    e_addr = f"127.0.0.1:{get_open_port()}"
    pd_addr = f"127.0.0.1:{get_open_port()}"

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--worker-addr",
        f"{e_addr}",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "10000",
        "--max-num-batched-tokens",
        "10000",
        "--worker-addr",
        f"{pd_addr}",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        f"{mooncake_ip}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "10000",
        "--eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
        "--metrics_port",
        str(metrics_port),
    ]

    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        pd_num=1,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as server:
        print("vllm instance is ready")
        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=[e_addr],
            pd_addr_list=[pd_addr],
            model_name=model,
            enable_health_monitor=True,
        )
        print("proxy is success")
        try:
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(max_tokens=128, temperature=0.0),
                request_id=str(uuid.uuid4()),
            )
            async for o in outputs:
                print(f"{o.outputs}", flush=True)
            p.shutdown()
        except Exception:
            assert server.check_log("invalid literal", 120), "init success"


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.function
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_lm_service_request_timeout_seconds_005(
    model: str, tp_size: int, dataset_name: str
):
    """
    LM_SERVICE_REQUEST_TIMEOUT_SECONDS is 1; single-node EPD, max_tokens=2000, serial
    generate calls fail and return an error.
    """
    image = convert_image_mode(
        Image.open(
            Path(__file__).parent.parent.parent.parent / "tools" / "224.png"
        ),
        "RGB",
    )
    IMAGE_ARRAY = np.array(image)
    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "1",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"127.0.0.1:{get_open_port()}"
    e_addr = f"127.0.0.1:{get_open_port()}"
    pd_addr = f"127.0.0.1:{get_open_port()}"

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--worker-addr",
        f"{e_addr}",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "10000",
        "--max-num-batched-tokens",
        "10000",
        "--worker-addr",
        f"{pd_addr}",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        f"{mooncake_ip}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "10000",
        "--eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
        "--metrics_port",
        str(metrics_port),
    ]

    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        pd_num=1,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as server:
        print("vllm instance is ready")
        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=[e_addr],
            pd_addr_list=[pd_addr],
            model_name=model,
            enable_health_monitor=True,
        )
        print("proxy is success")
        n_call = 3
        for call_num in range(n_call):
            print(f"**************call_num: {call_num}")
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=2000, temperature=0.0
                ),
                request_id=str(uuid.uuid4()),
            )
            async for o in outputs:
                print(f"{o.outputs}", flush=True)
            assert server.check_log("timed out after 1s", 120), "init success"
        p.shutdown()


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.function
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_lm_service_request_timeout_seconds_006(
    model: str, tp_size: int, dataset_name: str
):
    """
    LM_SERVICE_REQUEST_TIMEOUT_SECONDS is 1; multi-node EPD, max_tokens=2000, serial
    generate calls fail and return an error.
    """
    image = convert_image_mode(
        Image.open(
            Path(__file__).parent.parent.parent.parent / "tools" / "224.png"
        ),
        "RGB",
    )
    IMAGE_ARRAY = np.array(image)
    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "1",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(p_num):
        cluster.add_node_info("p", 2, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 3, CONTAINER_NAME)

    node_ips = get_cluster_ips()

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env(
            "p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    for i in range(d_num):
        env_dict.add_env(
            "d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i
        )

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"{node_ips[0]}:{get_open_port()}"
    e_addr = f"{node_ips[1]}:{get_open_port()}"
    p_addr = f"{node_ips[2]}:{get_open_port()}"
    d_addr = f"{node_ips[3]}:{get_open_port()}"

    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--worker-addr",
        f"{e_addr}",
        "--proxy-addr",
        f"{proxy_addr}",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "10000",
            "--max-num-batched-tokens",
            "10000",
            "--worker-addr",
            f"{p_addr}",
            "--proxy-addr",
            f"{proxy_addr}",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[2]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}',
        ],
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "10000",
            "--max-num-batched-tokens",
            "10000",
            "--worker-addr",
            f"{d_addr}",
            "--proxy-addr",
            f"{proxy_addr}",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[3]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}',
        ],
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        f"{mooncake_ip}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "10000",
        "--eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
        "--metrics_port",
        str(metrics_port),
    ]

    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        pd_num=1,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as server:
        print("vllm instance is ready")
        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=[e_addr],
            pd_addr_list=[p_addr, d_addr],
            model_name=model,
            enable_health_monitor=True,
        )
        print("proxy is success")
        n_call = 3
        for call_num in range(n_call):
            print(f"**************call_num: {call_num}")
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=2000, temperature=0.0
                ),
                request_id=str(uuid.uuid4()),
            )
            async for o in outputs:
                print(f"{o.outputs}", flush=True)
            assert server.check_log("timed out after 1s", 120), "init success"
        p.shutdown()


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.function
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_lm_service_request_timeout_seconds_007(
    model: str, tp_size: int, dataset_name: str
):
    """
    LM_SERVICE_REQUEST_TIMEOUT_SECONDS is 1; multi-node EPD, max_tokens=2000, parallel
    generate calls fail and return an error.
    """
    image = convert_image_mode(
        Image.open(
            Path(__file__).parent.parent.parent.parent / "tools" / "224.png"
        ),
        "RGB",
    )
    IMAGE_ARRAY = np.array(image)
    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "1",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(p_num):
        cluster.add_node_info("p", 2, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 3, CONTAINER_NAME)

    node_ips = get_cluster_ips()

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env(
            "p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    for i in range(d_num):
        env_dict.add_env(
            "d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i
        )

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"{node_ips[0]}:{get_open_port()}"
    e_addr = f"{node_ips[1]}:{get_open_port()}"
    p_addr = f"{node_ips[2]}:{get_open_port()}"
    d_addr = f"{node_ips[3]}:{get_open_port()}"

    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--worker-addr",
        f"{e_addr}",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "10000",
            "--max-num-batched-tokens",
            "10000",
            "--worker-addr",
            f"{p_addr}",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[2]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}',
        ],
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "10000",
            "--max-num-batched-tokens",
            "10000",
            "--worker-addr",
            f"{d_addr}",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[3]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}',
        ],
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        f"{mooncake_ip}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "10000",
        "--eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
        "--metrics_port",
        str(metrics_port),
    ]
    # proxy_args = ["--proxy-addr",f"{proxy_addr}","--worker-addr",f"{e_addr},{pd_addr}"]

    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        pd_num=1,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as server:
        print("vllm instance is ready")
        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=[e_addr],
            pd_addr_list=[p_addr, d_addr],
            model_name=model,
            enable_health_monitor=True,
        )
        print("proxy is success")
        n_call = 3

        # Create concurrent generate tasks
        tasks = []
        for call_num in range(n_call):
            print(f"**************Creating task for call_num: {call_num}")

            # Distinct request_id per call
            request_id = str(uuid.uuid4())

            # Spawn async task
            task = asyncio.create_task(
                p.generate(
                    prompt={
                        "prompt": PROMPT_TEMPLATE,
                        "multi_modal_data": {"image": IMAGE_ARRAY},
                    },
                    sampling_params=SamplingParams(
                        max_tokens=2000, temperature=0.0
                    ),
                    request_id=request_id,
                )
            )

            # Wrap task to consume generator outputs
            async def process_output(task_num, gen_task):
                outputs = await gen_task
                async for o in outputs:
                    print(f"{o.outputs}", flush=True)
                assert server.check_log("timed out after 1s", 120), (
                    "init success"
                )

            tasks.append(process_output(call_num, task))

        # Await all tasks concurrently
        print("Starting parallel execution...")
        await asyncio.gather(*tasks)
        p.shutdown()
