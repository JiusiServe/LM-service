# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import os

try:
    import pytest
    import socket
except ImportError:
    pass

from ..conftest import RemoteEPDServer
from .conftest import load_config

try:
    from tools.aisbench import run_aisbench_cases
    from vllm.utils import get_open_port
    from ..e2e.nightly.multi_node.config.utils import get_cluster_ips
except (ImportError, ModuleNotFoundError):
    pass
from ..nightly.multi_node.config.multi_node_epd_config import (
    ClusterManager,
    EnvManager,
)

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
CONTAINER_NAME = load_config().get("container_name")
DATASET_PATH = load_config().get("dataset_path")

TENSOR_PARALLELS = [1]
PREFIX_CACHE = [True, False]

REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1pd_mooncake_ipc_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    1E1PD, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: ipc
    """
    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "ipc",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "image_4"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY1E1PD_MOONCAKE_IPC",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1pdmerge_mooncake_ipc_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    1E1PD shared-card, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: ipc
    """

    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "ipc",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate,
            "result_file_name": f"{dataset}_PROXY1E1PDMERGE_MOONCAKE_IPC",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e2pd_mooncake_ipc_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    1E2PD, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: ipc
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "ipc",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY1E2PD_MOONCAKE_IPC",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e2pd_mooncake_tcp_ipv4_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    1E2PD, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv4)
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY1E2PD_MOONCAKE_TCP_IPV4",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e2pd_mooncake_tcp_ipv6_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    1E2PD, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv6)
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "::1"
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY1E2PD_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth_samereq"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("enable_prefix", PREFIX_CACHE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e2pd_mooncake_tcp_ipv6_002(
    model: str,
    tp_size: int,
    dataset: str,
    request_rate: float,
    enable_prefix: bool,
):
    """
    1E2PD, single-node deployment
    Prefix cache: on/off
    Dataset: same image within request
    ec transfer: mooncake
    Communication: tcp(ipv6)
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "::1"
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    if not enable_prefix:
        pd_arg.append("--no-enable-prefix-caching")
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY1E2PD_PREFIX_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("router", ROUTER)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy3e5pd_mooncake_tcp_ipv6_001(
    model: str, tp_size: int, dataset: str, request_rate: float, router: str
):
    """
    3E5PD, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Routing: RandomRouter, RoundRobinRouter, LeastInFlightRouter
    Communication: tcp(ipv6)
    """

    e_num = 3
    pd_num = 5
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "::1"
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    proxy_args = ["--router", router]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY3E5PD_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        proxy_args=proxy_args,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p1d_mooncake_ipc_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E1P1D, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth, image4
    ec transfer: mooncake
    Communication: ipc
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "ipc",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

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
        "20000",
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
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY1E1P1D_MOONCAKE_IPC",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy2e3p3d_mooncake_tcp_ipv6_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P2E3P3D, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: ipv6
    """

    e_num = 2
    p_num = 3
    d_num = 3
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "0.0.0.0"

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
        "20000",
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
    pd_server_args = list()

    p_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
        "--kv-transfer-config",
        f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
        f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
        '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
        f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
    ]
    d_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--kv-transfer-config",
        f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
        f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
        '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
        f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
    ]
    for _ in range(p_num):
        pd_server_args.append(p_arg)
    for _ in range(d_num):
        pd_server_args.append(d_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY2E3P3D_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth_samereq"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("enable_prefix", PREFIX_CACHE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p1d_mooncake_ipc_002(
    model: str,
    tp_size: int,
    dataset: str,
    request_rate: float,
    enable_prefix: bool,
):
    """
    P1E1P-1D, single-node deployment
    Prefix cache: on/off
    Dataset: same image within request
    ec transfer: mooncake
    Communication: ipc
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "ipc",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

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
        "20000",
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
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
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
    if not enable_prefix:
        for args in pd_server_args:
            args.append("--no-enable-prefix-caching")

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY1E1P1D_MOONCAKE_IPC",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p1d_mooncake_tcp_ipv4_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E1P-1D, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth, image4
    ec transfer: mooncake
    Communication: ipv4
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

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
        "20000",
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
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY1E1P1D_MOONCAKE_TCP_IPV4",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p1d_mooncake_tcp_ipv6_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E1P-1D, single-node deployment
    Prefix cache: on
    Dataset: simulate_truth, image4
    ec transfer: mooncake
    Communication: ipv6
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

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

    mooncake_ip = "::1"
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
        ],
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY1E1P1D_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy_1e_2pd_cross_mooncake_tcp_ipv4_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    1E2PD, multi-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv4)
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips()

    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")
    env_dict.add_env("pd", "MC_TCP_BIND_ADDRESS", f"{node_ips[2]}")

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 2, CONTAINER_NAME)

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY_1E_2PD_MOONCAKE_TCP_IPV4",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth_samereq"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("enable_prefix", PREFIX_CACHE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy_1e_2pd_cross_mooncake_tcp_ipv4_002(
    model: str,
    tp_size: int,
    dataset: str,
    request_rate: float,
    enable_prefix: bool,
):
    """
    1E2PD, multi-node deployment
    Prefix cache: on/off
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv4)
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips()
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")
    env_dict.add_env("pd", "MC_TCP_BIND_ADDRESS", f"{node_ips[2]}")

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 2, CONTAINER_NAME)

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)
    if not enable_prefix:
        pd_server_args.append("--no-enable-prefix-caching")

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY_1E_2PD_PREFIX_MOONCAKE_TCP_IPV4",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e_2pd_cross_mooncake_tcp_ipv6_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E——2PD, multi-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv6)
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips(family=socket.AF_INET6)
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("pd", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    cluster = ClusterManager()
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, CONTAINER_NAME)

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY1E_2PD_CROSS_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p_1d_cross_mooncake_tcp_ipv6_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E1P-1D, multi-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv6)
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips(family=socket.AF_INET6)
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("p", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("d", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")

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

    cluster = ClusterManager()
    for i in range(d_num):
        cluster.add_node_info("d", 1, CONTAINER_NAME)

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
        ],
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY1E1P_1D_CROSS_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e_1p_1d_cross_mooncake_tcp_ipv4_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E-1P-1D, multi-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv4)
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips()
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("p", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")
    env_dict.add_env("d", "MC_TCP_BIND_ADDRESS", f"{node_ips[2]}")

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

    cluster = ClusterManager()
    for i in range(d_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 2, CONTAINER_NAME)

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[2]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY1E_1P_1D_CROSS_MOONCAKE_TCP_IPV4",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("router", ROUTER)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy2e3p_3d_cross_mooncake_tcp_ipv6_001(
    model: str, tp_size: int, dataset: str, request_rate: float, router: str
):
    """
    P2E3P-3D, multi-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Routing: RandomRouter, RoundRobinRouter, LeastInFlightRouter
    Communication: tcp(ipv6)
    """

    e_num = 2
    p_num = 3
    d_num = 3
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips(family=socket.AF_INET6)
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("p", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("d", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")

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

    cluster = ClusterManager()
    for i in range(d_num):
        cluster.add_node_info("d", 1, CONTAINER_NAME)

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]
    pd_server_args = list()

    p_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
        "--kv-transfer-config",
        f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[0]}", '
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
        f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
        '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
        f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
    ]
    d_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--kv-transfer-config",
        f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
        f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
        '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
        f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
    ]
    for _ in range(p_num):
        pd_server_args.append(p_arg)
    for _ in range(d_num):
        pd_server_args.append(d_arg)

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

    proxy_args = ["--router", router]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "performance",
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY2E3P_3D_CROSS_MOONCAKE_TCP_IPV6",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        proxy_args=proxy_args,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test perf
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )


DATASET_NAME = ["textvqa_subset"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e2pd_mooncake_ipc_acc_001(
    model: str, tp_size: int, dataset: str
):
    """
    1E2PD, single-node deployment
    Prefix cache: on
    Dataset: textvqa_subset
    ec transfer: mooncake
    Communication: ipc
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "ipc",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "accuracy",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "request_conf": "vllm_api_general_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 2048,
            "max_out_len": 2048,
            "batch_size": 128,
            "temperature": 0,
            "top_k": -1,
            "top_p": 1,
            "repetition_penalty": 1,
            "request_rate": 0,
            "baseline": 81,
            "seed": 77,
            "threshold": 1,
        }
    ]
    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test acc
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            save=False,
        )


DATASET_NAME = ["textvqa_subset"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e2pd_mooncake_tcp_ipv4_acc_001(
    model: str, tp_size: int, dataset: str
):
    """
    1E2PD, single-node deployment
    Prefix cache: on
    Dataset: textvqa_subset
    ec transfer: mooncake
    Communication: ipv4
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
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

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "accuracy",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "request_conf": "vllm_api_general_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 2048,
            "max_out_len": 2048,
            "batch_size": 128,
            "temperature": 0,
            "top_k": -1,
            "top_p": 1,
            "repetition_penalty": 1,
            "request_rate": 0,
            "baseline": 81,
            "seed": 77,
            "threshold": 1,
        }
    ]
    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test acc
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            save=False,
        )


DATASET_NAME = ["textvqa_subset"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p1d_mooncake_tcp_ipv4_acc_001(
    model: str, tp_size: int, dataset: str
):
    """
    P1E1P1D, single-node deployment
    Prefix cache: on
    Dataset: textvqa_subset
    ec transfer: mooncake
    Communication: ipv4
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

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
        "20000",
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
        [
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "accuracy",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "request_conf": "vllm_api_general_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 2048,
            "max_out_len": 2048,
            "batch_size": 128,
            "temperature": 0,
            "top_k": -1,
            "top_p": 1,
            "repetition_penalty": 1,
            "request_rate": 0,
            "baseline": 81,
            "seed": 77,
            "threshold": 1,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test acc
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            save=False,
        )


DATASET_NAME = ["textvqa_subset"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p1d_mooncake_tcp_ipv6_acc_001(
    model: str, tp_size: int, dataset: str
):
    """
    P1E1P1D, single-node deployment
    Prefix cache: on
    Dataset: textvqa_subset
    ec transfer: mooncake
    Communication: ipv4
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

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

    mooncake_ip = "::1"
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
        ],
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "accuracy",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "request_conf": "vllm_api_general_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 2048,
            "max_out_len": 2048,
            "batch_size": 128,
            "temperature": 0,
            "top_k": -1,
            "top_p": 1,
            "repetition_penalty": 1,
            "request_rate": 0,
            "baseline": 81,
            "seed": 77,
            "threshold": 1,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test acc
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            save=False,
        )


DATASET_NAME = ["textvqa_subset"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy_1e_2pd_cross_mooncake_tcp_ipv4_acc_001(
    model: str, tp_size: int, dataset: str
):
    """
    P-1E-2PD, multi-node deployment
    Prefix cache: on
    Dataset: textvqa_subset
    ec transfer: mooncake
    Communication: ipv4
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    node_ips = get_cluster_ips()
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")
    env_dict.add_env("pd", "MC_TCP_BIND_ADDRESS", f"{node_ips[2]}")

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 2, CONTAINER_NAME)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "accuracy",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "request_conf": "vllm_api_general_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 2048,
            "max_out_len": 2048,
            "batch_size": 128,
            "temperature": 0,
            "top_k": -1,
            "top_p": 1,
            "repetition_penalty": 1,
            "request_rate": 0,
            "baseline": 81,
            "seed": 77,
            "threshold": 1,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test acc
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            save=False,
        )


DATASET_NAME = ["textvqa_subset"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e1p_1d_cross_mooncake_tcp_ipv6_acc_001(
    model: str, tp_size: int, dataset: str
):
    """
    P1E1P-1D, multi-node deployment
    Prefix cache: on
    Dataset: textvqa_subset
    ec transfer: mooncake
    Communication: ipv6
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    node_ips = get_cluster_ips(family=socket.AF_INET6)
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("p", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("d", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")

    cluster = ClusterManager()
    for i in range(d_num):
        cluster.add_node_info("d", 1, CONTAINER_NAME)

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
        ],
    ]

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "accuracy",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "request_conf": "vllm_api_general_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 2048,
            "max_out_len": 2048,
            "batch_size": 128,
            "temperature": 0,
            "top_k": -1,
            "top_p": 1,
            "repetition_penalty": 1,
            "request_rate": 0,
            "baseline": 81,
            "seed": 77,
            "threshold": 1,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test acc
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            save=False,
        )


REQUEST_RATE = [0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.stability
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e_2pd_cross_mooncake_tcp_ipv6_stability_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E——2PD, multi-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv6)
    """

    e_num = 1
    pd_num = 2
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips()
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("pd", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )

    cluster = ClusterManager()
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, CONTAINER_NAME)

    pd_server_args = list()

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_arg = [
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--max-model-len",
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout": "20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]
    for _ in range(pd_num):
        pd_server_args.append(pd_arg)

    mooncake_args = [
        "--rpc_port",
        str(rpc_port),
        "--rpc_address",
        "::",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]
    aisbench_cases = [
        {
            "case_type": "pressure",
            "pressure_time": 86400,
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + pd_num),
            "result_file_name": f"{dataset}_PROXY1E_2PD_CROSS_MOONCAKE_TCP_IPV6_STABILITY",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=pd_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test stability
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + pd_num,
        )


REQUEST_RATE = [0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.stability
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset", DATASET_NAME)
async def test_proxy1e_1p_1d_cross_mooncake_tcp_ipv4_stability_001(
    model: str, tp_size: int, dataset: str, request_rate: float
):
    """
    P1E-1P-1D, multi-node deployment
    Prefix cache: on
    Dataset: simulate_truth
    ec transfer: mooncake
    Communication: tcp(ipv4)
    """

    e_num = 1
    p_num = 1
    d_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_dict = EnvManager()
    node_ips = get_cluster_ips()
    env_dict.add_env("common", env_dict=env)
    env_dict.add_env("proxy", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("e", "MC_TCP_BIND_ADDRESS", f"{node_ips[0]}")
    env_dict.add_env("p", "MC_TCP_BIND_ADDRESS", f"{node_ips[1]}")
    env_dict.add_env("d", "MC_TCP_BIND_ADDRESS", f"{node_ips[2]}")

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

    cluster = ClusterManager()
    for i in range(d_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 2, CONTAINER_NAME)

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
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
        "20000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
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
            "20000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[2]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
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

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(
                DATASET_PATH, "simulate_truth_samereq"
            ),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    aisbench_cases = [
        {
            "case_type": "pressure",
            "pressure_time": 86400,
            "request_conf": "vllm_api_stream_chat",
            "dataset_path": os.path.join(DATASET_PATH, dataset),
            "dataset_conf": "textvqa/textvqa_gen_base64",
            "num_prompts": 200,
            "batch_size": 128,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": request_rate * (e_num + p_num + d_num),
            "result_file_name": f"{dataset}_PROXY1E_1P_1D_CROSS_MOONCAKE_TCP_IPV4_STABILITY",
            "baseline": 1,
            "seed": 77,
            "threshold": 0.97,
        }
    ]

    api_port = 10002
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="mooncake",
        kv_store_type="mooncake",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=p_num + d_num,
        e_num=e_num,
        env_dict=env_dict,
        node_info=cluster,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # test stability
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=aisbench_cases,
            card_num=e_num + p_num + d_num,
        )
