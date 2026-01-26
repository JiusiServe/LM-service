# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import asyncio
import contextlib
import gc
import json
import os
import re
import copy
import subprocess
import threading
import traceback
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union, Literal

import numpy as np
import pandas as pd
import psutil
import pytest
import requests
import torch
import importlib
from PIL import Image

from modelscope import snapshot_download  # type: ignore[import-untyped]
from torch import nn
from transformers import (
    BatchEncoding,
    BatchFeature,
)


# TODO: remove this part after the patch merged into vllm, if
# we not explicitly patch here, some of them might be effectiveless
# in pytest scenario

from vllm.utils import get_open_port

from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment,
    destroy_model_parallel,
)

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)
_M = TypeVar("_M")

_PromptMultiModalInput = Union[List[_M], List[List[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[Tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]

_TEST_DIR = os.path.dirname(__file__)
DISAGG_EPD_PROXY_SCRIPT = "../../../examples/epd/disagg_epd_proxy.py"


def get_package_location(package_name):
    try:
        distribution = importlib.metadata.distribution(package_name)
        return str(distribution.locate_file(""))
    except importlib.metadata.PackageNotFoundError:
        return None


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def write_to_execl(data, path):
    if path is not None:
        if not os.path.exists(path):
            df = pd.DataFrame(data, index=[0])
            df.to_csv(path, index=False)
        else:
            existing_df = pd.read_csv(path)
            new_df = pd.DataFrame(data, index=[0])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(path, index=False)


class SharedInfoManager:
    def __init__(self):
        self._metrics = {}
        self._e_addr_list = []
        self._pd_addr_list = []
        self._p_addr_list = []
        self._d_addr_list = []
        self._enable_ttft_breakdown = False
        self._check_info = ""
        self._lock = threading.Lock()

    def update_metrics(self, metrics):
        with self._lock:
            self._metrics = metrics

    def get_metrics(self):
        with self._lock:
            return self._metrics.copy()

    def open_breakdown(self):
        with self._lock:
            self._enable_ttft_breakdown = True

    def get_breakdown_status(self):
        with self._lock:
            return self._enable_ttft_breakdown

    def update_check_info(self, check_info):
        with self._lock:
            self._check_info = check_info

    def get_check_info(self):
        with self._lock:
            return self._check_info

    def add_addr_list(self, addr, role):
        with self._lock:
            if role.lower() == "e":
                self._e_addr_list.append(addr)
            elif role.lower() == "p":
                self._p_addr_list.append(addr)
            elif role.lower() == "d":
                self._d_addr_list.append(addr)
            else:
                self._pd_addr_list.append(addr)

    def get_addr_list(self, role):
        with self._lock:
            if role.lower() == "e":
                return self._e_addr_list.copy()
            elif role.lower() == "p":
                return self._p_addr_list.copy()
            elif role.lower() == "d":
                return self._d_addr_list.copy()
            else:
                return self._pd_addr_list.copy()


class OutputManager:
    def __init__(self, share_info: SharedInfoManager):
        self.info = share_info
        self._text = ""
        self._check_result = False

    def read_output(self, pipe, prefix):
        try:
            with pipe:
                for line in iter(pipe.readline, ""):
                    if line:
                        self._text = line
                        print(f"{prefix}: {line}", end="")
                        if self.info.get_breakdown_status():
                            self._extract_ttft_data(line, prefix)
                        if not self.info.get_check_info():
                            self._check_result = self._check_assign_info(
                                line, self.info.get_check_info()
                            )

        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc()

    def _check_assign_info(self, text, check_info):
        if check_info in text:
            return True
        else:
            return False

    def get_check_result(self):
        return self._check_result

    def _extract_ttft_data(self, text, prefix):
        metrics = self.info.get_metrics()
        if "PROXY" in prefix.upper():
            patterns = {}
            for i, flag in enumerate(self.info.get_addr_list("e")):
                patterns[f"E{i}_queue"] = (
                    rf"{flag}.*Avg queue time requests: ([\d.]+) ms"
                )
                patterns[f"E{i}_prefill"] = (
                    rf"{flag}.*Avg prefill time requests: ([\d.]+) ms"
                )
                patterns["transfer_to_encode"] = (
                    rf"{flag}.*Avg proxy to instance requests time: ([\d.]+) ms"
                )
            for i, flag in enumerate(self.info.get_addr_list("pd")):
                patterns[f"PD{i}_ttft"] = (
                    rf"{flag}.*Avg proxy ttft: ([\d.]+) ms"
                )
                patterns[f"PD{i}_queue"] = (
                    rf"{flag}.*Avg queue time requests: ([\d.]+) ms"
                )
                patterns[f"PD{i}_prefill"] = (
                    rf"{flag}.*Avg prefill time requests: ([\d.]+) ms"
                )
                patterns[f"PD{i}_first_token"] = (
                    rf"{flag}.*Avg time to first token: ([\d.]+) ms"
                )
                patterns["transfer_to_pd"] = (
                    rf"{flag}.*Avg proxy to instance requests time: ([\d.]+) ms"
                )
            for key, pattern in patterns.items():
                match = re.search(pattern, text)
                if match:
                    metrics[key] = float(match.group(1))
        self.info.update_metrics(metrics)


class ContainerManager:
    def __init__(self, output: OutputManager):
        self._container_processes = []
        self._output = output

    def run_in_remote_container(
        self,
        host,
        container_name,
        server_cmd: list[str],
        env_dict: Optional[dict[str, str]],
        log_prefix: str,
    ) -> None:
        host = host.replace("[", "").replace("]", "")
        docker_cmd = ["docker", "exec", "-i"]

        if env_dict:
            for key, value in env_dict.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        docker_cmd.append(container_name)
        for i in range(3, len(server_cmd)):
            arg = server_cmd[i]
            if arg.startswith("{") and arg.endswith("}"):
                server_cmd[i] = f"'{arg}'"
        docker_cmd.extend(server_cmd)
        ssh_cmd = ["ssh", f"root@{host}"] + docker_cmd
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        stdout_thread = threading.Thread(
            target=self._output.read_output,
            args=(proc.stdout, log_prefix),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._output.read_output,
            args=(proc.stderr, log_prefix),
            daemon=True,
        )

        stdout_thread.start()
        stderr_thread.start()
        process_info = {
            "ssh_proc": proc,
            "host": host,
            "container_name": container_name,
        }
        self._container_processes.append(process_info)

    def kill_container_process_only(self):
        for process_info in self._container_processes:
            ssh_proc = process_info["ssh_proc"]
            container_name = process_info["container_name"]
            host = process_info["host"]

            try:
                kill_cmd = [
                    "ssh",
                    f"root@{host}",
                    "docker",
                    "exec",
                    container_name,
                    "pkill",
                    "-f",
                    "-TERM",
                    "python",  # Send TERM signal first
                ]
                subprocess.run(kill_cmd, timeout=10, capture_output=True)
                time.sleep(10)
                kill_cmd_force = [
                    "ssh",
                    f"root@{host}",
                    "docker",
                    "exec",
                    container_name,
                    "pkill",
                    "-f",
                    "-KILL",
                    "python",  # Force kill
                ]
                subprocess.run(kill_cmd_force, timeout=10, capture_output=True)
                kill_cmd_force = [
                    "ssh",
                    f"root@{host}",
                    "docker",
                    "exec",
                    container_name,
                    "pkill",
                    "-f",
                    "-KILL",
                    "VLLM",  # Force kill
                ]
                subprocess.run(kill_cmd_force, timeout=10, capture_output=True)

                print(
                    f"container process tree in {container_name} of {host} is killed"
                )

            except Exception as e:
                print(
                    f"kill process tree in {container_name} of {host} failed: {e}"
                )

            try:
                ssh_proc.terminate()
                ssh_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ssh_proc.kill()


def run_server_new_session(
    server_cmd: list[str],
    env_dict: Optional[dict[str, str]],
    log_prefix: str,
    output: OutputManager,
) -> None:
    """Subclasses override this method to customize server process launch"""
    env = os.environ.copy()
    # the current process might initialize npu,
    # to be safe, we should use spawn method
    if env_dict is not None:
        env.update(env_dict)
    proc = subprocess.Popen(
        server_cmd,
        cwd=None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    stdout_thread = threading.Thread(
        target=output.read_output, args=(proc.stdout, log_prefix), daemon=True
    )
    stderr_thread = threading.Thread(
        target=output.read_output, args=(proc.stderr, log_prefix), daemon=True
    )

    stdout_thread.start()
    stderr_thread.start()
    return proc


def kill_process_tree(pid):
    """kill process and its children"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        gone, still_alive = psutil.wait_procs(children, timeout=10)

        for child in still_alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        try:
            parent.terminate()
            parent.wait(timeout=10)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass

    except psutil.NoSuchProcess:
        pass


class RemoteEPDServer:
    def get_proxy(self):
        return self.p

    def check_log(self, check_info, timeout) -> bool:
        self._share_info.update_check_info(check_info)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._output.get_check_result():
                return True
        return False

    def save_ttft_data(self, file_name, index):
        data = {"index": index}
        data.update(self._share_info.get_metrics())
        write_to_execl(data, f"./{file_name}.csv")
        print(f"TTFT Analysis csv file is locate in ./{file_name}.csv")

    def _delete_shm(self) -> None:
        for i, arg in enumerate(
            self.e_serve_args_list + self.pd_serve_args_list
        ):
            if "--ec-transfer-config" in arg:
                index = arg.index("--ec-transfer-config")
                shm_path = (
                    json.loads(arg[index + 1])
                    .get("ec_connector_extra_config")
                    .get("shared_storage_path")
                )
                args = ["rm", "-r", "-f", shm_path]
                print(f"delete shm_path is: {shm_path}")
                self._run_server(args, None, "[DELETE] ")

        if (
            self.node_info is not None
            and self.node_info.get_node_info("e") is not None
        ):
            for i in range(1, len(self.cluster_ips)):
                self._container.run_in_remote_container(
                    host=self.cluster_ips[i],
                    container_name=self.node_info.get_node_info(
                        "e"
                    ).container_name,
                    server_cmd=["rm", "-r", "-f", shm_path],
                    env_dict=None,
                    log_prefix="[DELETE] ",
                )

    def _run_server(
        self,
        server_cmd: list[str],
        env_dict: Optional[dict[str, str]],
        log_prefix: str,
    ) -> None:
        """Subclasses override this method to customize server process launch"""
        env = os.environ.copy()
        # the current process might initialize npu,
        # to be safe, we should use spawn method
        if env_dict is not None:
            env.update(env_dict)
        proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # text mode
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=self._output.read_output,
            args=(proc.stdout, log_prefix),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._output.read_output,
            args=(proc.stderr, log_prefix),
            daemon=True,
        )

        stdout_thread.start()
        stderr_thread.start()
        self._proc_list.append(proc)

    def _start_api_server(self) -> None:
        common_env = self.env_dict.get_node_env("common", 0)
        role_env = self.env_dict.get_node_env("proxy", 0)
        env = (
            {**common_env, **role_env}
            if common_env is not None and role_env is not None
            else common_env
            if common_env is not None
            else role_env
        )
        api_server_args = [
            "--host",
            "127.0.0.1",
            "--port",
            str(self.api_server_port),
            "--proxy-config",
            json.dumps(self.proxy_config),
        ]

        print(f"proxy params is: {api_server_args}")
        api_server_path = (
            Path(__file__).parent.parent.parent / "tools" / "api_server.py"
        )
        api_server_args = ["python", api_server_path, *api_server_args]
        self._proc_list.append(
            run_server_new_session(
                api_server_args, env, "[PROXY] ", self._output
            )
        )

    def _start_mooncake(self) -> None:
        mooncake_args_list = list()
        if isinstance(self.mooncake_args, list):
            if not all(isinstance(item, list) for item in self.mooncake_args):
                mooncake_args_list.append(copy.deepcopy(self.mooncake_args))
            else:
                mooncake_args_list = self.mooncake_args
        else:
            raise RuntimeError("mooncake_args must be a list")
        for arg in mooncake_args_list:
            mooncake_arg = ["mooncake_master", *arg]
            self._proc_list.append(
                run_server_new_session(
                    mooncake_arg,
                    self.env_dict.get_node_env("common", 0),
                    "[MOONCAKE] ",
                    self._output,
                )
            )

    def _start_etcd(self) -> None:
        etcd_client_port = get_open_port()
        etcd_peer_port = get_open_port()
        host = self.cluster_ips[0]
        if self.enable_ipv6:
            etcd_args = [
                "etcd",
                "--name",
                "etcd-epd",
                "--data-dir",
                "/tmp/etcd-epd-data",
                "--listen-client-urls",
                f"http://[::]:{etcd_client_port}",
                "--advertise-client-urls",
                f"http://{host}:{etcd_client_port}",
                "--listen-peer-urls",
                f"http://[::]:{etcd_peer_port}",
                "--initial-advertise-peer-urls",
                f"http://{host}:{etcd_peer_port}",
                "--initial-cluster",
                f"etcd-epd=http://{host}:{etcd_peer_port}",
                "--initial-cluster-token",
                "etcd-cluster-ipv6",
                "--initial-cluster-state",
                "new",
            ]
        else:
            etcd_args = [
                "etcd",
                "--name",
                "etcd-epd",
                "--data-dir",
                "/tmp/etcd-epd-data",
                "--listen-client-urls",
                f"http://{host}:{etcd_client_port}",
                "--advertise-client-urls",
                f"http://{host}:{etcd_client_port}",
                "--listen-peer-urls",
                f"http://{host}:{etcd_peer_port}",
                "--initial-advertise-peer-urls",
                f"http://{host}:{etcd_peer_port}",
                "--initial-cluster",
                f"etcd-epd=http://{host}:{etcd_peer_port}",
            ]

        self.etcd_address = f"{host}:{etcd_client_port}"
        run_server_new_session(
            ["rm", "-rf", "/tmp/etcd-epd-data"], None, "[ETCD] ", self._output
        )
        self._proc_list.append(
            run_server_new_session(etcd_args, None, "[ETCD] ", self._output)
        )

    def _start_datasystem(self) -> None:
        self.env_dict.add_env("common", "EC_STORE_TYPE", "datasystem")
        self.env_dict.add_env("common", "USING_PREFIX_CONNECTOR", "0")
        self.datasystem_port = get_open_port()
        run_server_new_session(
            ["rm", "-rf", "datasystem"], None, "[DATASYSTEM_0] ", self._output
        )
        self._proc_list.append(
            run_server_new_session(
                [
                    "dscli",
                    "start",
                    "-w",
                    "--worker_address",
                    f"{self.cluster_ips[0]}:{self.datasystem_port}",
                    "--etcd_address",
                    self.etcd_address,
                    "--shared_memory_size_mb",
                    "200000",
                ],
                None,
                "[DATASYSTEM_0] ",
                self._output,
            )
        )

        if self.node_info is not None:
            for i in range(1, len(self.cluster_ips)):
                self._container.run_in_remote_container(
                    host=self.cluster_ips[i],
                    container_name=self.node_info.get_node_info(
                        "ds", i - 1
                    ).container_name,
                    server_cmd=["rm", "-rf", "datasystem"],
                    env_dict=None,
                    log_prefix=f"[DATASYSTEM_{i}] ",
                )
                self._container.run_in_remote_container(
                    host=self.cluster_ips[i],
                    container_name=self.node_info.get_node_info(
                        "ds", i - 1
                    ).container_name,
                    server_cmd=[
                        "dscli",
                        "start",
                        "-w",
                        "--worker_address",
                        f"{self.cluster_ips[i]}:{self.datasystem_port}",
                        "--etcd_address",
                        self.etcd_address,
                        "--shared_memory_size_mb",
                        "200000",
                    ],
                    env_dict=None,
                    log_prefix=f"[DATASYSTEM_{i}] ",
                )
        time.sleep(10)

    def _stop_datasystem(self) -> None:
        self._proc_list.append(
            run_server_new_session(
                [
                    "dscli",
                    "stop",
                    "--worker_address",
                    f"{self.cluster_ips[0]}:{self.datasystem_port}",
                ],
                None,
                "[DATASYSTEM_0] ",
                self._output,
            )
        )

        if self.node_info is not None:
            for i in range(1, len(self.cluster_ips)):
                self._container.run_in_remote_container(
                    host=self.cluster_ips[i],
                    container_name=self.node_info.get_node_info(
                        "ds", i - 1
                    ).container_name,
                    server_cmd=[
                        "dscli",
                        "stop",
                        "--worker_address",
                        f"{self.cluster_ips[i]}:{self.datasystem_port}",
                    ],
                    env_dict=None,
                    log_prefix=f"[DATASYSTEM_{i}] ",
                )

    def _get_addr_config(self, args, i, role):
        if (
            common_env := self.env_dict.get_node_env("common", 0)
        ) and common_env.get("TRANSFER_PROTOCOL") is not None:
            self.protocol = self.env_dict.get_node_env("common", 0)[
                "TRANSFER_PROTOCOL"
            ].lower()
        elif "--transfer-protocol" in args:
            protocol_index = args.index("--transfer-protocol") + 1
            if protocol_index < len(args):
                self.protocol = args[protocol_index].lower()
        else:
            self.protocol = "ipc"

        if self.protocol == "tcp":
            if self.node_info is not None:
                if self.node_info.get_node_info(role.lower()) is not None:
                    node_id = self.node_info.get_node_info(
                        role.lower(), i
                    ).node_id
                    host = self.cluster_ips[node_id]
                else:
                    host = self.cluster_ips[0]
            else:
                host = self.cluster_ips[0]
            return {
                "proxy_addr": f"{self.cluster_ips[0]}:{self.proxy_port}",
                "worker_addr": f"{host}:{get_open_port()}",
            }
        else:
            if role.lower() == "e":
                return {
                    "proxy_addr": f"{self._default_addr_prefix}proxy",
                    "worker_addr": f"{self._default_addr_prefix}encoder_{i}",
                }
            elif role.lower() == "p":
                return {
                    "proxy_addr": f"{self._default_addr_prefix}proxy",
                    "worker_addr": f"{self._default_addr_prefix}p_{i}",
                }
            else:
                return {
                    "proxy_addr": f"{self._default_addr_prefix}proxy",
                    "worker_addr": f"{self._default_addr_prefix}pd_{i}",
                }

    def _start_vllm_worker(self):
        self.env_dict.add_env("common", "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        self.env_dict.add_env("common", "VLLM_USE_V1", "1")
        self.env_dict.add_env("common", "VLLM_NIXL_SIDE_CHANNEL_PORT", "6000")
        self.env_dict.add_env(
            "common", "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
        )

        serve_arg_cmd = [
            "taskset",
            "-c",
            "0-96",
            "python",
            "-m",
            "lm_service.entrypoints.worker",
        ]

        for i, e_serve_arg in enumerate(self.e_serve_args_list):
            e_serve_arg = [*serve_arg_cmd, *e_serve_arg]

            config = self._get_addr_config(e_serve_arg, i, "E")
            if "--proxy-addr" not in e_serve_arg:
                e_serve_arg.extend(["--proxy-addr", config["proxy_addr"]])
            if "--worker-addr" not in e_serve_arg:
                e_serve_arg.extend(["--worker-addr", config["worker_addr"]])

            index_e = e_serve_arg.index("--proxy-addr")
            if (
                self.proxy_addr is not None
                and e_serve_arg[index_e + 1] != self.proxy_addr
            ):
                raise ValueError("proxy addr must be same between workers")
            self.proxy_addr = e_serve_arg[index_e + 1]

            index_e = e_serve_arg.index("--worker-addr")
            self._share_info.add_addr_list(e_serve_arg[index_e + 1], "e")

            if "--model" not in e_serve_arg:
                raise ValueError("must carry --model")
            else:
                index_e = e_serve_arg.index("--model")
                if (
                    self.model is not None
                    and e_serve_arg[index_e + 1] != self.model
                ):
                    raise ValueError("model must be same between workers")
                self.model = e_serve_arg[index_e + 1]

            common_env = self.env_dict.get_node_env("common", 0)
            role_env = self.env_dict.get_node_env("e", i)
            env = (
                {**common_env, **role_env}
                if common_env is not None and role_env is not None
                else common_env
                if common_env is not None
                else role_env
            )
            if (
                self.node_info is not None
                and self.node_info.get_node_info("e") is not None
            ):
                node_id = self.node_info.get_node_info("e", i).node_id
                if self.datasystem_port is not None:
                    env.update(
                        {
                            "DS_WORKER_ADDR": f"{self.cluster_ips[node_id]}:{self.datasystem_port}"
                        }
                    )

                self._container.run_in_remote_container(
                    host=self.cluster_ips[node_id],
                    container_name=self.node_info.get_node_info(
                        "e", i
                    ).container_name,
                    server_cmd=e_serve_arg,
                    env_dict=env,
                    log_prefix=f"[ENCODE_{i}] ",
                )
            else:
                if self.datasystem_port is not None:
                    env.update(
                        {
                            "DS_WORKER_ADDR": f"{self.cluster_ips[0]}:{self.datasystem_port}"
                        }
                    )
                self._run_server(e_serve_arg, env, f"[ENCODE_{i}] ")

        current_p_num = -1
        current_d_num = -1
        for i, pd_serve_arg in enumerate(self.pd_serve_args_list):
            pd_serve_arg = [*serve_arg_cmd, *pd_serve_arg]
            if "--model" not in pd_serve_arg:
                raise ValueError("must carry --model")
            else:
                index_pd = pd_serve_arg.index("--model")
                if (
                    self.model is not None
                    and pd_serve_arg[index_pd + 1] != self.model
                ):
                    raise ValueError("model must be same between workers")

            log_prefix = ""
            role = ""
            current_node_index = 0
            if "--kv-transfer-config" in pd_serve_arg:
                kv_index = pd_serve_arg.index("--kv-transfer-config")
                if "kv_consumer" in pd_serve_arg[kv_index + 1]:
                    current_d_num += 1
                    log_prefix = f"[D_{current_d_num}] "
                    role = "d"
                    current_node_index = current_d_num
                elif "kv_producer" in pd_serve_arg[kv_index + 1]:
                    current_p_num += 1
                    log_prefix = f"[P_{current_p_num}] "
                    role = "p"
                    current_node_index = current_p_num
            else:
                log_prefix = f"[PD_{i}] "
                role = "pd"
                current_node_index = i

            config = self._get_addr_config(
                pd_serve_arg, current_node_index, role
            )
            if "--proxy-addr" not in pd_serve_arg:
                pd_serve_arg.extend(["--proxy-addr", config["proxy_addr"]])
            if "--worker-addr" not in pd_serve_arg:
                pd_serve_arg.extend(["--worker-addr", config["worker_addr"]])

            index_pd = pd_serve_arg.index("--proxy-addr")
            if (
                self.proxy_addr is not None
                and pd_serve_arg[index_pd + 1] != self.proxy_addr
            ):
                raise ValueError("proxy addr must be same between workers")

            worker_index = pd_serve_arg.index("--worker-addr")
            self._share_info.add_addr_list(pd_serve_arg[worker_index + 1], role)

            common_env = self.env_dict.get_node_env("common", 0)
            role_env = self.env_dict.get_node_env(role, current_node_index)
            env = (
                {**common_env, **role_env}
                if common_env is not None and role_env is not None
                else common_env
                if common_env is not None
                else role_env
            )

            if (
                self.node_info is not None
                and self.node_info.get_node_info(role) is not None
            ):
                node_id = self.node_info.get_node_info(
                    role, current_node_index
                ).node_id
                if self.datasystem_port is not None:
                    env.update(
                        {
                            "DS_WORKER_ADDR": f"{self.cluster_ips[node_id]}:{self.datasystem_port}"
                        }
                    )
                self._container.run_in_remote_container(
                    host=self.cluster_ips[node_id],
                    container_name=self.node_info.get_node_info(
                        role, current_node_index
                    ).container_name,
                    server_cmd=pd_serve_arg,
                    env_dict=env,
                    log_prefix=log_prefix,
                )
            else:
                if self.datasystem_port is not None:
                    env.update(
                        {
                            "DS_WORKER_ADDR": f"{self.cluster_ips[0]}:{self.datasystem_port}"
                        }
                    )
                self._run_server(pd_serve_arg, env, log_prefix)

    def _start_zmq_proxy(self):
        from lm_service.apis.vllm.proxy import Proxy
        from lm_service.routing_logic import (
            RandomRouter,
            RoundRobinRouter,
            LeastInFlightRouter,
        )

        if self.env_dict.get_node_env("common", 0) is not None:
            for key, value in self.env_dict.get_node_env("common", 0).items():
                os.environ[key] = value

        if self.env_dict.get_node_env("proxy", 0) is not None:
            for key, value in self.env_dict.get_node_env("proxy", 0).items():
                os.environ[key] = value

        self.proxy_config = {"model_name": self.model}
        if (
            self.proxy_args is not None
            and "--metastore-client-config" in self.proxy_args
        ):
            self.proxy_config["metastore_client_config"] = json.loads(
                self.proxy_args[
                    self.proxy_args.index("--metastore-client-config") + 1
                ]
            )

        self.proxy_config.update(
            {
                "proxy_addr": self.proxy_addr,
                "encode_addr_list": self._share_info.get_addr_list("e"),
            }
        )
        if self._share_info.get_addr_list("pd"):
            self.proxy_config["pd_addr_list"] = self._share_info.get_addr_list(
                "pd"
            )
        else:
            self.proxy_config.update(
                {
                    "p_addr_list": self._share_info.get_addr_list("p"),
                    "d_addr_list": self._share_info.get_addr_list("d"),
                }
            )
        if (
            self.proxy_args is not None
            and "--transfer-protocol" in self.proxy_args
        ):
            self.proxy_config["transfer_protocol"] = self.proxy_args[
                self.proxy_args.index("--transfer-protocol") + 1
            ]
        if (
            self.proxy_args is not None
            and "--enable-health-monitor" in self.proxy_args
        ):
            self.proxy_config["enable_health_monitor"] = self.proxy_args[
                self.proxy_args.index("--enable-health-monitor") + 1
            ]
        if (
            self.proxy_args is not None
            and "--health-check-interval" in self.proxy_args
        ):
            self.proxy_config["health_check_interval"] = self.proxy_args[
                self.proxy_args.index("--health-check-interval") + 1
            ]
        if self.proxy_args is not None and "--router" in self.proxy_args:
            self.proxy_config["router"] = self.proxy_args[
                self.proxy_args.index("--router") + 1
            ]
            if (
                self.proxy_args[self.proxy_args.index("--router") + 1]
                == "RandomRouter"
            ):
                self.proxy_config["router"] = RandomRouter
            elif (
                self.proxy_args[self.proxy_args.index("--router") + 1]
                == "RoundRobinRouter"
            ):
                self.proxy_config["router"] = RoundRobinRouter
            else:
                self.proxy_config["router"] = LeastInFlightRouter

        print(f"proxy params is: {self.proxy_config}")
        p = Proxy(**self.proxy_config)
        if self.proxy_args is not None and "--router" in self.proxy_args:
            self.proxy_config["router"] = self.proxy_args[
                self.proxy_args.index("--router") + 1
            ]
        return p

    def _start_vllm_serve(self):
        self.env_dict.add_env("common", "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        self.env_dict.add_env("common", "VLLM_USE_V1", "1")
        self.env_dict.add_env(
            "common", "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
        )

        serve_arg_cmd = ["taskset", "-c", "0-96", "vllm", "serve"]

        common_env = self.env_dict.get_node_env("common", 0)

        for i, e_serve_arg in enumerate(self.e_serve_args_list):
            role_env = self.env_dict.get_node_env("e", i)
            env = (
                {**common_env, **role_env}
                if common_env is not None and role_env is not None
                else common_env
                if common_env is not None
                else role_env
            )
            e_serve_arg = [*serve_arg_cmd, *e_serve_arg]
            e_port = get_open_port()
            if "--port" not in e_serve_arg:
                e_serve_arg.extend(["--port", str(e_port)])
            index_e = e_serve_arg.index("--port")
            self._share_info.add_addr_list(
                f"http://localhost:{e_serve_arg[index_e + 1]}", "e"
            )
            self._run_server(e_serve_arg, env, f"[ENCODE_{i}] ")

        current_d_num = -1
        current_p_num = -1
        for i, pd_serve_arg in enumerate(self.pd_serve_args_list):
            if "--kv-transfer-config" in pd_serve_arg:
                kv_index = pd_serve_arg.index("--kv-transfer-config")
                if "kv_consumer" in pd_serve_arg[kv_index + 1]:
                    current_d_num += 1
                    log_prefix = f"[D_{current_d_num}] "
                    role = "d"
                    current_node_index = current_d_num
                elif "kv_producer" in pd_serve_arg[kv_index + 1]:
                    current_p_num += 1
                    log_prefix = f"[P_{current_p_num}] "
                    role = "p"
                    current_node_index = current_p_num
            else:
                log_prefix = f"[PD_{i}] "
                role = "pd"
                current_node_index = i

            role_env = self.env_dict.get_node_env(role, current_node_index)
            env = (
                {**common_env, **role_env}
                if common_env is not None and role_env is not None
                else common_env
                if common_env is not None
                else role_env
            )
            pd_serve_arg = [*serve_arg_cmd, *pd_serve_arg]
            pd_port = get_open_port()
            if "--port" not in pd_serve_arg:
                pd_serve_arg.extend(["--port", str(pd_port)])
            index_pd = pd_serve_arg.index("--port")
            self._share_info.add_addr_list(
                f"http://localhost:{pd_serve_arg[index_pd + 1]}", role
            )
            self._run_server(pd_serve_arg, env, log_prefix)

    async def _wait_for_vllm_worker(self, max_wait_seconds) -> None:
        from lm_service.protocol.protocol import ServerType

        sleep_times = 10
        timeout_times = 3
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            tasks_0 = [
                asyncio.create_task(
                    asyncio.wait_for(
                        self.p.check_health(
                            ServerType.E_INSTANCE, f"{self.protocol}://{addr}"
                        ),
                        timeout=timeout_times,
                    )
                )
                for addr in self._share_info.get_addr_list("e")
            ]
            if self._share_info.get_addr_list("pd"):
                tasks_1 = [
                    asyncio.create_task(
                        asyncio.wait_for(
                            self.p.check_health(
                                ServerType.PD_INSTANCE,
                                f"{self.protocol}://{addr}",
                            ),
                            timeout=timeout_times,
                        )
                    )
                    for addr in self._share_info.get_addr_list("pd")
                ]
                tasks = tasks_0 + tasks_1
            else:
                tasks_1 = [
                    asyncio.create_task(
                        asyncio.wait_for(
                            self.p.check_health(
                                ServerType.P_INSTANCE,
                                f"{self.protocol}://{addr}",
                            ),
                            timeout=timeout_times,
                        )
                    )
                    for addr in self._share_info.get_addr_list("p")
                ]
                tasks_2 = [
                    asyncio.create_task(
                        asyncio.wait_for(
                            self.p.check_health(
                                ServerType.D_INSTANCE,
                                f"{self.protocol}://{addr}",
                            ),
                            timeout=timeout_times,
                        )
                    )
                    for addr in self._share_info.get_addr_list("d")
                ]
                tasks = tasks_0 + tasks_1 + tasks_2

            results = await asyncio.gather(*tasks, return_exceptions=True)
            if all([isinstance(result, bool) and result for result in results]):
                print("All instances are ready")
                return
            else:
                print(f"current results: {results}")
                await asyncio.sleep(sleep_times)

        raise RuntimeError("epd instance start failed!")

    async def _wait_for_server(
        self,
        port,
        host="127.0.0.1",
        timeout: int = 300,
        check_interval: float = 0.5,
    ) -> bool:
        base_url = f"http://{host}:{port}"
        health_url = f"{base_url}/health"

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=3)
                if response.status_code == 200:
                    return True
                else:
                    print(
                        f"❌ server start error, http error code: {response.status_code}"
                    )
            except requests.exceptions.ConnectionError:
                print("⏳ waiting for ready ...")
            except requests.exceptions.RequestException as e:
                print(f"server start error: {e}")

            await asyncio.sleep(check_interval)
        print("server start timeout")
        return False

    def __init__(
        self,
        run_mode: Literal["serve", "worker"],
        store_type: Literal["mooncake", "storage", "datasystem"],
        e_num: Optional[int],
        pd_num: Optional[int],
        e_serve_args,
        pd_serve_args,
        proxy_type: Literal["proxy", "api_server"] = None,
        kv_store_type: Literal["mooncake", "datasystem"] = "",
        mooncake_args=None,
        proxy_args: Union[list[str], str] = None,
        api_server_port: Optional[int] = 10001,
        env_dict=None,
    ) -> None:
        self._share_info = SharedInfoManager()
        self._output = OutputManager(self._share_info)
        self._container = ContainerManager(self._output)
        self._proc_list = list()
        self.e_num = e_num
        self.pd_num = pd_num
        self.p = None
        if run_mode not in ["serve", "worker"]:
            raise ValueError("run mode must be serve or worker")
        if store_type not in ["mooncake", "storage", "datasystem"]:
            raise ValueError("store type must be mooncake or storage")
        if kv_store_type not in ["mooncake", "datasystem", ""]:
            raise ValueError("kv store type must be mooncake")
        if proxy_type is not None and proxy_type not in ["proxy", "api_server"]:
            raise ValueError("proxy type must be proxy or api_server")
        self.run_mode = run_mode
        self.store_type = store_type
        self.protocol = ""
        self.kv_store_type = kv_store_type
        self.proxy_type = proxy_type
        self.api_server_port = api_server_port
        self.e_serve_args_list = list()
        self.pd_serve_args_list = list()
        self.proxy_port = get_open_port()
        self.model = None
        self.datasystem_port = None
        if isinstance(e_serve_args, list):
            if not all(isinstance(item, list) for item in e_serve_args):
                for i in range(self.e_num):
                    self.e_serve_args_list.append(copy.deepcopy(e_serve_args))
            else:
                self.e_serve_args_list = e_serve_args
        else:
            raise RuntimeError("e_serve_args must be a list")

        if isinstance(pd_serve_args, list):
            if not all(isinstance(item, list) for item in pd_serve_args):
                for i in range(self.pd_num):
                    self.pd_serve_args_list.append(copy.deepcopy(pd_serve_args))
            else:
                self.pd_serve_args_list = pd_serve_args
        else:
            raise RuntimeError("pd_serve_args must be a list")

        self.mooncake_args = mooncake_args
        self.proxy_args = proxy_args
        self.env_dict = env_dict
        if (
            common_env := self.env_dict.get_node_env("common", 0)
        ) and common_env.get("TIMECOUNT_ENABLED", 0) == "1":
            self._share_info.open_breakdown()
        self._default_addr_prefix = "/tmp/"
        self.proxy_addr = None
        if (
            common_env := self.env_dict.get_node_env("common", 0)
        ) and common_env.get("MC_USE_IPV6", "") == "1":
            self.enable_ipv6 = True
        else:
            self.enable_ipv6 = False

    async def __aenter__(self):
        # start with
        max_wait_seconds = 600
        if self.store_type == "mooncake" or self.kv_store_type == "mooncake":
            self._start_mooncake()
        if (
            self.store_type == "datasystem"
            or self.kv_store_type == "datasystem"
        ):
            self._start_etcd()
            self._start_datasystem()
        if self.store_type == "storage":
            self._delete_shm()
        if self.run_mode == "worker":
            self._start_vllm_worker()
            self.p = self._start_zmq_proxy()
            await self._wait_for_vllm_worker(max_wait_seconds=max_wait_seconds)
        elif self.run_mode == "serve":
            self._start_vllm_serve()
            for url in self._share_info.get_addr_list("e"):
                port = url.split(":")[-1]
                await self._wait_for_server(port=port)
            for url in self._share_info.get_addr_list("pd"):
                port = url.split(":")[-1]
                await self._wait_for_server(port=port)
        if self.proxy_type is None and self.p is not None:
            self.p.shutdown()
        elif self.proxy_type == "api_server":
            self.p.shutdown()
            self._start_api_server()
            await self._wait_for_server(port=self.api_server_port)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # exit with
        if (
            self.store_type == "datasystem"
            or self.kv_store_type == "datasystem"
        ):
            self._stop_datasystem()

        for proc in self._proc_list:
            kill_process_tree(proc.pid)
        self._container.kill_container_process_only()
        print("vllm instance and api server is stopping")
        if self.p is not None:
            self.p.shutdown()
        print("proxy is stopping")


class DisaggEpdProxy:
    def _start_disagg_proxy(self):
        proxy_args = [
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--encode-servers-urls",
            ",".join(self.server._share_info.get_addr_list("e")),
        ]

        if self.server._share_info.get_addr_list("pd"):
            proxy_args.extend(
                [
                    "--decode-servers-urls",
                    ",".join(self.server._share_info.get_addr_list("pd")),
                    "--prefill-servers-urls",
                    "disable",
                ]
            )
        else:
            proxy_args.extend(
                [
                    "--decode-servers-urls",
                    ",".join(self.server._share_info.get_addr_list("d")),
                    "--prefill-servers-urls",
                    ",".join(self.server._share_info.get_addr_list("p")),
                ]
            )

        print(f"proxy param is: {proxy_args}")
        proxy_args = ["python", DISAGG_EPD_PROXY_SCRIPT, *proxy_args]
        self._proc_list.append(
            run_server_new_session(
                proxy_args, self.env_dict, "[PRXOY] ", self.server._output
            )
        )

    async def _wait_for_server(
        self, timeout: int = 300, check_interval: float = 0.5
    ) -> bool:
        base_url = f"http://127.0.0.1:{self.port}"
        health_url = f"{base_url}/health"

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=3)
                if response.status_code == 200:
                    print("✅proxy is ready")
                    return True
                else:
                    print(
                        f"❌ proxy start error, http error code: {response.status_code}"
                    )
            except requests.exceptions.ConnectionError:
                print("⏳ waiting for ready ...")
            except requests.exceptions.RequestException as e:
                print(f"proxy start error: {e}")

            await asyncio.sleep(check_interval)
        print("proxy start timeout")
        return False

    def __init__(
        self,
        port,
        proxy_args: Union[list[str], str] = None,
        env_dict=None,
        server: RemoteEPDServer = None,
    ) -> None:
        self.port = port
        self.proxy_args = proxy_args
        self.server = server
        self.env_dict = env_dict
        self._proc_list = list()

    async def __aenter__(self):
        self._start_disagg_proxy()
        await self._wait_for_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # exit with
        for proc in self._proc_list:
            kill_process_tree(proc.pid)
        print("proxy is stopping")


@pytest.fixture(scope="session")
def ilama_lora_files():
    return snapshot_download(repo_id="vllm-ascend/ilama-text2sql-spider")


def qwen_prompt(questions: List[str]) -> List[str]:
    placeholder = "<|image_pad|>"
    return [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{q}<|im_end|>\n<|im_start|>assistant\n"
        )
        for q in questions
    ]


PROMPT_TEMPLATES = {
    "qwen2.5vl": qwen_prompt,
}


@pytest.fixture(params=list(PROMPT_TEMPLATES.keys()))
def prompt_template(request):
    return PROMPT_TEMPLATES[request.param]
