# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import regex as re


def is_addr_ipv6(addr: str | None) -> bool:
    """
    Check if the given address is an IPv6 address

    Args:
        addr (str): The address to check

    Returns:
        bool: True if the address is an IPv6 address, False otherwise
    """
    if addr is None:
        return False
    # Support addresses with protocol prefix like "tcp://[::1]:8090"
    # Match protocol prefix (optional), IPv6 address in square brackets, and port (optional)
    ipv6_pattern = r"^(?:[a-zA-Z0-9]+://)?\[([0-9a-fA-F:]+)\](?::(\d+))?$"
    return bool(re.match(ipv6_pattern, addr))


def get_heartbeat_addr(addr: str) -> str:
    """
    Get the heartbeat address based on the worker address.

    Args:
        addr: The worker address (e.g., "tcp://127.0.0.1:8000" or "ipc:///tmp/worker.ipc")

    Returns:
        The heartbeat address.
    """
    if "://" not in addr:
        # Assume it's an IP:Port string if no protocol
        protocol = "tcp"
        base_addr = addr
    else:
        protocol, base_addr = addr.split("://", 1)

    if protocol == "ipc":
        return f"ipc://{base_addr}_hb"
    elif protocol == "tcp":
        if is_addr_ipv6(addr):
            # IPv6 logic
            match = re.match(r"^\[([0-9a-fA-F:]+)\](?::(\d+))?$", base_addr)
            if match:
                ip, port = match.groups()
                new_port = int(port) + 1000  # Shift port by 1000 for HB
                return f"tcp://[{ip}]:{new_port}"
        else:
            # IPv4 logic
            if ":" in base_addr:
                ip, port = base_addr.rsplit(":", 1)
                new_port = int(port) + 1000  # Shift port by 1000 for HB
                return f"tcp://{ip}:{new_port}"

    # Fallback/Error case, return unmodified (might fail but better than crash)
    return f"{addr}_hb"
