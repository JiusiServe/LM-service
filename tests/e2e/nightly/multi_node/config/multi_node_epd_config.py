from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class NodeConfig:
    node_id: int
    container_name: str

@dataclass
class ClusterManager:
    node_info: Dict[str, List[NodeConfig]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_info:
            self.node_info = {
                "e": [],
                "pd": [],
                "p": [],
                "d": [],
                "ds": []
            }

    def add_node_info(self, node_type: str, node_id: int, container_name="epd_vllm_ascend"):
        if node_type not in self.node_info:
            raise ValueError("node type can only be e,pd,p,d,ds")
        new_config = NodeConfig(node_id=node_id, container_name=container_name)
        self.node_info[node_type].append(new_config)
        print(f"add {node_type}: node_id={node_id}, container={container_name}")


    def get_all_info(self):
        return self.node_info


    def get_node_info(self, node_type: str, index: int = 0) -> Optional[NodeConfig]:
        if node_type in self.node_info and index < len(self.node_info[node_type]):
            return self.node_info[node_type][index]
        return None


@dataclass
class EnvManager:
    env_info: Dict[str, List[dict]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.env_info:
            self.env_info = {
                "e": [],
                "pd": [],
                "p": [],
                "d": [],
                "proxy": [],
                "common": []
            }

    def add_env(self, node_type: str, env_key: str = "", env_value: str = "", env_dict=None, index=0):
        if node_type not in self.env_info:
            raise ValueError("node type can only be e,pd,p,d,proxy,common")
        if env_dict is not None:
            env_list = list()
            index_list = list()
            if not isinstance(env_dict, list):
                env_list.append(env_dict)
            if not isinstance(index, list):
                index_list.append(index)
            for env, index in zip(env_list, index_list):
                if index >= len(self.env_info[node_type]):
                    self.env_info[node_type].append(env)
                else:
                    self.env_info[node_type][index].update(env)
        else:
            if index >= len(self.env_info[node_type]):
                self.env_info[node_type].append({env_key: env_value})
            else:
                self.env_info[node_type][index][env_key] = env_value



    def get_all_env(self):
        return self.env_info


    def get_node_env(self, node_type: str, index: int = None):
        if node_type in self.env_info and index is None:
            return self.env_info[node_type]
        elif node_type in self.env_info and index < len(self.env_info[node_type]):
            return self.env_info[node_type][index]
        return None
