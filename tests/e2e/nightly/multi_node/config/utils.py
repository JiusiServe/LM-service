import logging
import os
import socket
from contextlib import contextmanager
from typing import Optional

import psutil

# import torch.distributed as dist


@contextmanager
def temp_env(env_dict):
    old_env = {}
    for k, v in env_dict.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# @contextmanager
# def dist_group(backend="gloo"):
#     if dist.is_initialized():
#         yield
#         return

#     dist.init_process_group(backend=backend)
#     try:
#         yield
#     finally:
#         dist.destroy_process_group()

def get_cluster_ips(family=socket.AF_INET) -> list[str]:
    if os.getenv("CLUSTER_ADDRESS"):
        cluster_dns = os.getenv("CLUSTER_ADDRESS").split(";")
        result_ips = list()
        try:
            for dns in cluster_dns:
                result = socket.getaddrinfo(dns, None, family)
                result_ips.append(result[0][4][0])
            return result_ips
        except Exception as e:
            print(f"cluster ip load failed: {e}, will use default set")
            return []
    else:
        return []


def get_avaliable_port(start_port: int = 6000, end_port: int = 7000) -> int:
    import socket
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


def get_cur_ip() -> str:
    """Returns the current machine's IP address."""
    return socket.gethostbyname_ex(socket.gethostname())[2][0]


def get_net_interface(ip: Optional[str] = None) -> Optional[str]:
    """
    Returns specified IP's inetwork interface.
    If no IP is provided, uses the first from hostname -I.
    """
    if ip is None:
        ip = get_cur_ip()

    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == ip:
                return iface
    return None


def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
