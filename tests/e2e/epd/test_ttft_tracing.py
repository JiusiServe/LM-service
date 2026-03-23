# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import os
import copy
import pytest

from ..conftest import RemoteEPDServer
from .conftest import load_config
from ..nightly.multi_node.config.multi_node_epd_config import EnvManager

try:
    from tools.aisbench import run_aisbench_cases
except (ImportError, ModuleNotFoundError):
    pass

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")

TENSOR_PARALLELS = [1]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_timeout_enabled_001(model: str, tp_size: int):
    """timeout_enabled=1: instance starts successfully and prints timing statistics."""
    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")
    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "1")
    env_dict.add_env("common", "TIMECOUNT_ENABLED", "1")
    env_dict.add_env("common", "VLLM_LOG_STATS_INTERVAL", "10")

    e_server_args = [
        "--no-enable-prefix-caching",
        "--model",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
    ]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen",
            "num_prompts": 100,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0.0,
            "seed": 77,
        }
    ]

    request_rate = [0.28]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold": 0.97,
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10003
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="storage",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=2,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # aisbench test
        run_aisbench_cases(
            model=model, port=api_port, aisbench_cases=aisbench_cases
        )


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_timeout_enabled_002(model: str, tp_size: int):
    """timeout_enabled is neither 0 nor 1: instance start fails and prints the error."""
    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")
    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "1")
    env_dict.add_env("common", "TIMECOUNT_ENABLED", "0.5")
    env_dict.add_env("common", "VLLM_LOG_STATS_INTERVAL", "10")

    e_server_args = [
        "--no-enable-prefix-caching",
        "--model",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
    ]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen",
            "num_prompts": 100,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0.0,
            "seed": 77,
        }
    ]

    request_rate = [0.28]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold": 0.97,
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10003
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="storage",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=2,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # aisbench test
        run_aisbench_cases(
            model=model, port=api_port, aisbench_cases=aisbench_cases
        )


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_vllm_log_stats_interval_001(model: str, tp_size: int):
    """vllm_log_stats_interval=20: instance starts successfully and prints timing statistics."""
    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")
    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "1")
    env_dict.add_env("common", "TIMECOUNT_ENABLED", "1")
    env_dict.add_env("common", "VLLM_LOG_STATS_INTERVAL", "20")

    e_server_args = [
        "--no-enable-prefix-caching",
        "--model",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
    ]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen",
            "num_prompts": 100,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0.0,
            "seed": 77,
        }
    ]

    request_rate = [0.28]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold": 0.97,
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10003
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="storage",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=2,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # aisbench test
        run_aisbench_cases(
            model=model, port=api_port, aisbench_cases=aisbench_cases
        )


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_performance_001(model: str, tp_size: int):
    """Timing enabled; QPS 0.28-1.78; instance starts and prints stats; compare TTFT/TPOT vs timing off for regression."""
    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")
    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "1")
    env_dict.add_env("common", "TIMECOUNT_ENABLED", "1")
    env_dict.add_env("common", "VLLM_LOG_STATS_INTERVAL", "10")
    e_server_args = [
        "--no-enable-prefix-caching",
        "--model",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
    ]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0.0,
            "seed": 77,
        }
    ]

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 100,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": "94_performance_001_open",
        "threshold": 0.97,
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10003
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="storage",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=2,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # aisbench test
        run_aisbench_cases(
            model=model, port=api_port, aisbench_cases=aisbench_cases
        )


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_performance_002(model: str, tp_size: int):
    """Timing disabled; QPS 0.28-1.78; instance starts successfully and prints timing statistics."""
    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")
    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "1")
    env_dict.add_env("common", "TIMECOUNT_ENABLED", "0")
    env_dict.add_env("common", "VLLM_LOG_STATS_INTERVAL", "10")
    e_server_args = [
        "--no-enable-prefix-caching",
        "--model",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
    ]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0.0,
            "seed": 77,
        }
    ]

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 100,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": "94_performance_001_close",
        "threshold": 0.97,
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10003
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="storage",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=2,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # aisbench test
        run_aisbench_cases(
            model=model, port=api_port, aisbench_cases=aisbench_cases
        )


@pytest.mark.skip(reason="skip for now")
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_longterm_001(model: str, tp_size: int):
    """Timing enabled; QPS 0.28; instance starts; sustained load 2h; prints timing statistics."""
    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env(
            "pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i
        )
    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")
    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "1")
    env_dict.add_env("common", "TIMECOUNT_ENABLED", "1")
    env_dict.add_env("common", "VLLM_LOG_STATS_INTERVAL", "10")
    e_server_args = [
        "--no-enable-prefix-caching",
        "--model",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        "--model",
        model,
        "--max-model-len",
        "30000",
        "--max-num-batched-tokens",
        "40000",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"'
        + SHARED_STORAGE_PATH
        + '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
    ]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen",
            "num_prompts": 100,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0.0,
            "seed": 77,
        }
    ]

    request_rate = [0.28]
    case_dict = {
        "case_type": "performance",
        "pressure_time": 86400,
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": "94_longterm_001",
        "threshold": 0.97,
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10003
    async with RemoteEPDServer(
        run_mode="worker",
        store_type="storage",
        proxy_type="api_server",
        api_server_port=api_port,
        pd_num=2,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )

        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=case_dict,
        )
