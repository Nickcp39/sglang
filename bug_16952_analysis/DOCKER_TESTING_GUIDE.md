# Bug #16952 Docker 测试指南

## 为什么必须使用 Docker？

### 1. DeepEP 依赖复杂
- **DeepEP** 是 DeepSeek 团队开发的通信库，安装过程非常繁琐
- 需要编译 NVSHMEM、GDRCopy、IBGDA 等多个底层库
- 官方 Docker 镜像已经预装并优化好了这些算子
- 本地环境安装 DeepEP 可能需要数小时，且容易出错

### 2. TBO (Two-batch Overlap) 依赖底层调度
- TBO 涉及多个 Stream 的同步和底层 CUDA 调度
- 需要正确的 NCCL 版本（Docker 镜像中已配置为 `nvidia-nccl-cu12==2.27.6`）
- 本地环境如果 NCCL 版本不对，可能会报一些莫名其妙的通信错误
- 让你分不清到底是 Bug 还是环境问题

### 3. 复现的一致性
- 用户报告 Bug 时通常是在高性能计算集群上
- 使用官方镜像能保证你遇到的"报错"和用户遇到的"报错"是同一个逻辑
- 避免因为环境差异导致的误判

## Docker 镜像选择

### 官方镜像
- **生产环境**: `lmsysorg/sglang:latest`
- **开发环境**: `lmsysorg/sglang:dev`（包含开发工具）

### 镜像特点
- ✅ 预装 DeepEP（已编译并优化）
- ✅ 预装正确的 NCCL 版本
- ✅ 预装所有 CUDA kernels
- ✅ 预装 FlashInfer with CuteDSL support
- ✅ 配置好 RDMA/InfiniBand 支持

## 快速开始

### 1. 拉取镜像

```bash
# 拉取最新镜像
docker pull lmsysorg/sglang:latest

# 或者拉取开发镜像（包含更多工具）
docker pull lmsysorg/sglang:dev
```

### 2. 运行测试（Windows PowerShell）

```powershell
# 进入测试目录
cd d:\Yc-software\github_code\sglang\bug_16952_analysis\test

# 运行 Docker 测试脚本
.\docker_test.ps1
```

### 3. 手动运行（Linux/Mac）

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 32123:32123 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --network=host \
    --privileged \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tp 2 \
        --ep 2 \
        --port 32123 \
        --trust-remote-code \
        --enable-dp-attention \
        --disable-custom-all-reduce \
        --enable-two-batch-overlap \
        --moe-a2a-backend deepep \
        --log-level info
```

## Windows 上的 Docker 注意事项

### 1. 网络模式
- Windows 上 `--network=host` 可能不工作
- 使用端口映射 `-p 32123:32123` 代替

### 2. RDMA 支持
- Windows Docker Desktop 不支持 RDMA
- 如果不需要 RDMA，可以移除 `--network=host --privileged`
- 但保留它们也不会有害

### 3. GPU 支持
- 需要安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 确保 Docker Desktop 中启用了 GPU 支持

### 4. 卷挂载路径
- Windows 路径格式：`C:\Users\username\.cache\huggingface`
- Docker 中需要使用 `/c/Users/username/.cache/huggingface` 或 `/mnt/c/Users/username/.cache/huggingface`

## 测试场景

### 场景 1: 复现 Bug（修复前）
```bash
# 应该看到 AssertionError
docker run --gpus all --shm-size 32g -p 32123:32123 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --network=host --privileged \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tp 2 --ep 2 --port 32123 \
        --trust-remote-code \
        --enable-dp-attention \
        --disable-custom-all-reduce \
        --enable-two-batch-overlap \
        --moe-a2a-backend deepep \
        --log-level info
```

### 场景 2: 验证修复（修复后）
```bash
# 应该看到自动选择日志，然后成功启动
docker run --gpus all --shm-size 32g -p 32123:32123 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --network=host --privileged \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tp 2 --ep 2 --port 32123 \
        --trust-remote-code \
        --enable-dp-attention \
        --disable-custom-all-reduce \
        --enable-two-batch-overlap \
        --moe-a2a-backend deepep \
        --log-level info
```

**预期日志**：
```
TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.
```

### 场景 3: 手动指定 backend（验证兼容性）
```bash
# 手动指定 flashinfer_cutedsl，应该也能工作
docker run --gpus all --shm-size 32g -p 32123:32123 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --network=host --privileged \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tp 2 --ep 2 --port 32123 \
        --trust-remote-code \
        --enable-dp-attention \
        --disable-custom-all-reduce \
        --enable-two-batch-overlap \
        --moe-a2a-backend deepep \
        --moe-runner-backend flashinfer_cutedsl \
        --log-level info
```

## 开发模式（挂载代码）

如果你想在 Docker 中测试本地代码修改：

```bash
# 挂载本地代码
docker run -itd --gpus all --shm-size 32g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /path/to/sglang:/sgl-workspace/sglang \
    --ipc=host --network=host --privileged \
    --name sglang_bug16952 \
    lmsysorg/sglang:dev /bin/zsh

# 进入容器
docker exec -it sglang_bug16952 /bin/zsh

# 在容器内重新安装（如果修改了代码）
cd /sgl-workspace/sglang
pip install -e "python[all]"
```

## 常见问题

### Q: Windows 上 Docker 无法使用 GPU？

**A**: 
1. 确保安装了 NVIDIA Container Toolkit
2. 确保 Docker Desktop 中启用了 GPU 支持
3. 检查 `docker run --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi` 是否工作

### Q: 模型下载很慢？

**A**: 
1. 使用 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
2. 挂载本地缓存：`-v ~/.cache/huggingface:/root/.cache/huggingface`
3. 在容器外先下载模型

### Q: 如何查看容器日志？

**A**: 
```bash
# 查看运行中的容器日志
docker logs -f <container_name>

# 查看已停止的容器日志
docker logs <container_name>
```

### Q: 如何进入运行中的容器调试？

**A**: 
```bash
docker exec -it <container_name> /bin/zsh
```

## RTX 4090 测试说明

### Qwen3-30B 无法在单卡 RTX 4090 上运行

**原因**：
- Qwen3-30B 需要约 60GB 显存（FP16/BF16）
- RTX 4090 只有 24GB 显存
- 用户配置 `--tp 8 --ep 8` 需要 8 个 GPU

### 替代方案：使用更小的 MoE 模型

**推荐模型**：`Qwen/Qwen1.5-MoE-A2.7B-Chat`
- 约 2.7B 参数，可以在 RTX 4090 上运行
- 使用 `--tp 2 --ep 2` 即可
- 同样可以复现 Bug

**测试命令**：
```bash
docker run --gpus all --shm-size 32g -p 32123:32123 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --network=host --privileged \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tp 2 --ep 2 --port 32123 \
        --trust-remote-code \
        --enable-dp-attention \
        --disable-custom-all-reduce \
        --enable-two-batch-overlap \
        --moe-a2a-backend deepep \
        --log-level info
```

## 复现 Bug 的步骤

### 场景 1: 复现 Bug（修复前）

使用用户的原始命令，**不添加** `--moe-runner-backend flashinfer_cutedsl`：

```bash
docker run --gpus all --shm-size 32g -p 32123:32123 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --network=host --privileged \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tp 2 --ep 2 --port 32123 \
        --trust-remote-code \
        --enable-dp-attention \
        --disable-custom-all-reduce \
        --enable-two-batch-overlap \
        --moe-a2a-backend deepep \
        --log-level info
```

**预期错误**：
```
AssertionError: forward_deepgemm_masked is deprecated
```

### 场景 2: 验证修复（修复后）

**预期日志**：
```
TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.
```

然后服务器应该能正常启动。

## 总结

**强烈建议使用 Docker**，因为：
1. ✅ DeepEP 安装复杂，Docker 已预装
2. ✅ NCCL 版本正确，避免通信错误
3. ✅ 复现一致性，与用户环境一致
4. ✅ 节省时间，避免环境配置问题
