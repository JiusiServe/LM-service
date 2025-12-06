# openYuanrong datasystem 快速使用指南

## 概述
openYuanrong datasystem 是一个分布式缓存系统，利用计算集群的 HBM/DRAM/SSD 资源构建近计算多级缓存，提升模型训练及推理、大数据、微服务等场景数据访问性能。

## 环境要求
- 操作系统：openEuler 22.03 或更高版本
- CANN：8.2.rc1 或更高版本
- Python：3.9–3.11
- etcd：3.5.12 或更高版本

## 部署 etcd

### 安装 etcd

#### 1. 下载二进制包 [etcd github releases](https://github.com/etcd-io/etcd/releases)
```bash
ETCD_VERSION="v3.5.12"  
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz
```
#### 2. 解压并安装
```bash
tar -xvf etcd-${ETCD_VERSION}-linux-amd64.tar.gz
cd etcd-${ETCD_VERSION}-linux-amd64
# copy the binary to system
sudo cp etcd etcdctl /usr/local/bin/
```
#### 3. 验证安装
```bash
etcd --version
etcdctl version
```

### 启动 etcd
> 提示: 此为最小化单节点部署示例。生产环境请参考 [etcd official site](https://etcd.io/docs/current/op-guide/clustering/).

#### 拉起单节点 etcd 集群
```bash
etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://0.0.0.0:2380 \
  --initial-cluster etcd-single=http://0.0.0.0:2380 &
```

#### 参数说明
- --name：cluster name
- --data-dir：directory to store data
- --listen-client-urls：address to listen from clients (0.0.0.0 allows access from any IP address)
- --advertise-client-urls：address advertised to clients
- --listen-peer-urls：address to listen from other nodes in the cluster
- --initial-advertise-peer-urls：address advertised to other nodes in the cluster
- --initial-cluster：initial nodes in the cluster (format: name1=peer_url1,name2=peer_url2,...)

#### 验证 etcd 是否正常运行
```bash
etcdctl --endpoints "127.0.0.1:2379" put key "value"
etcdctl --endpoints "127.0.0.1:2379" get key
```
若能成功写入并读取，表示 etcd 已正确部署。

## 部署 openYuanrong datasystem

### 安装 openYuanrong datasystem
推荐方式：通过 pip 安装预编译 wheel 包
```bash
pip install https://openyuanrong.obs.cn-southwest-2.myhuaweicloud.com/openyuanrong_datasystem-0.5.0-cp39-cp39-manylinux_2_34_x86_64.whl
```

### 启动 openYuanrong datasystem
安装 openYuanrong datasystem 后，即可通过随包自带的 dscli 命令行工具一键完成集群部署。

#### 简单示例
假设 etcd 地址为 10.170.27.165:2379：

节点 1
```bash
dscli start -w \
  --worker_address "10.170.27.165:31501" \
  --etcd_address "10.170.27.165:2379" \
  --shared_memory_size_mb 20000
```

节点 2
```bash
dscli start -w \
  --worker_address "10.170.27.163:31501" \
  --etcd_address "10.170.27.165:2379" \
  --shared_memory_size_mb 20000
```

节点 3
```bash
dscli start -w \
  --worker_address "10.170.27.161:31501" \
  --etcd_address "10.170.27.165:2379" \
  --shared_memory_size_mb 20000
```

### 一键卸载
在每个节点执行：
```bash
dscli stop --worker_address "127.0.0.1:31501"
```
替换 127.0.0.1:31501 为实际 worker 地址。

## 配置 VLLM 参数使用 Yuanrong Connector

### 使用 EC Connector
Datasystem 支持通过 ECMooncakeStorageConnector（用于 EC 传输）和 YuanRongConnector（用于 KV 传输）与 VLLM 对接。
```bash
export DS_WORKER_ADDR="127.0.0.1:31501"   # 替换为本机实际 worker 地址
export EC_STORE_TYPE="datasystem"
export USING_PREFIX_CONNECTOR=0
```

#### 1E1PD Encoder实例示例：
* 使用ipv4时，启动模型时添加如下配置：
$HOST_IP为本机IP， $MOONCAKE_MASTER_IP和$MOONCAKE_MASTER_PORT为mooncake master所在节点IP和port，
$MOONCAKE_METADATA_IP和$MOONCAKE_METADATA_PORT为元数据服务所在节点IP和port。
```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_producer",
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 32212254720,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```

#### 1E1PD PD实例示例：
```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_consumer",
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 0,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```
* 使用ipv6时，注意添加中括号：
```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_producer", # PD实例需要修改为ec_consumer
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://['$MOONCAKE_METADATA_IP']:'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 32212254720,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "['$MOONCAKE_MASTER_IP']:'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```

### 使用EC connector和KV connector
EC connector和mooncake一样
同 1E1PD 中的 Encoder 配置。
```bash
export DS_WORKER_ADDR="127.0.0.1:31501"   # 替换为本机实际 worker 地址
export EC_STORE_TYPE="datasystem"
export USING_PREFIX_CONNECTOR=0
```
kv_connector要配置为 YuanRongConnector

### 1E1P1D Encoder实例：
* 使用ipv4时，启动模型时添加如下配置：
$HOST_IP为本机IP， $MOONCAKE_MASTER_IP和$MOONCAKE_MASTER_PORT为mooncake master所在节点IP和port，
$MOONCAKE_METADATA_IP和$MOONCAKE_METADATA_PORT为元数据服务所在节点IP和port。
```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_producer",
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 32212254720,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```
### 1E1P1D P实例示例：
```bash
 --kv-transfer-config '{
    "kv_connector": "YuanRongConnector",
	"kv_role": "kv_producer"
 }'
 --ec-transfer-config '{
    "ec_connector":"ECMooncakeStorageConnector",
    "ec_role":"ec_consumer",
    "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 0,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1
   }
 }'
```
### 1E1P1D D实例示例：
```bash
 --kv-transfer-config '{
	"kv_connector": "YuanRongConnector",
	"kv_role": "kv_consumer",
 }'
```
