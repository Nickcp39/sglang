# Bug #16952 快速开始指南

## 如果你已经安装好 Docker

如果你已经安装好 Docker 并且之前成功运行过容器（比如在 port 30000），可以直接开始：

**注意路径**：
- 如果你在**项目根目录** (`d:\Yc-software\github_code\sglang`)，使用 `cd bug_16952_analysis\test`
- 如果你已经在 **`bug_16952_analysis` 目录**，使用 `cd test`

### 步骤 1: 快速检查（可选）

```powershell
# 检查 GPU 支持（如果之前运行过应该没问题）
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**如果正常显示 GPU 信息**，继续下一步。

### 步骤 2: 拉取镜像

```powershell
docker pull lmsysorg/sglang:latest
```

**等待**: 可能需要 5-15 分钟（取决于网络速度）

### 步骤 3: 测试修复前（复现 Bug）

**直接运行命令**（推荐，最简单）:

**单 GPU**（RTX 4090）:
```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

**多 GPU**（2 个或更多）:
```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

**或者使用脚本**（如果脚本能运行）:
```powershell
cd bug_16952_analysis\test
.\docker_test.ps1
```

**预期**: 应该看到错误 `AssertionError: forward_deepgemm_masked is deprecated`

**如果看到错误**，说明环境配置正确，继续下一步。

### 步骤 4: 应用修复

```powershell
# 回到项目根目录
cd d:\Yc-software\github_code\sglang

# 应用修复
git apply bug_16952_analysis\code\patch_server_args.py.patch
git apply bug_16952_analysis\code\patch_ep_moe_layer.py.patch
```

### 步骤 5: 测试修复后

**5.1 启动容器并挂载代码**:

```powershell
# 停止之前的容器（如果有）
docker stop sglang_bug16952 2>$null
docker rm sglang_bug16952 2>$null

# 启动容器
docker run -itd --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" -v "${PWD}:/sgl-workspace/sglang" --ipc=host --network=host --privileged --name sglang_bug16952 lmsysorg/sglang:latest /bin/bash
```

**5.2 在容器内重新安装 sglang**:

```powershell
docker exec sglang_bug16952 bash -c "cd /sgl-workspace/sglang && pip install -e 'python[all]'"
```

**5.3 运行测试**:

**单 GPU**:
```powershell
docker exec -it sglang_bug16952 python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

**多 GPU**:
```powershell
docker exec -it sglang_bug16952 python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

**或者使用脚本**:
```powershell
cd bug_16952_analysis\test
.\docker_test_with_fix.ps1
```

**预期**: 
- 应该看到日志 `TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl...`
- 然后服务器成功启动

---

## 完整流程

如果需要更详细的说明，请查看 **[WORKFLOW.md](./WORKFLOW.md)**

---

## 关于端口

**注意**: 
- 测试脚本使用 **port 32123**（不是 30000）
- 如果你之前的东西还在运行，不会冲突
- 如果需要更改端口，修改脚本中的 `$PORT = 32123`

---

## 常见问题

### Q: 看到 `CUDA error: invalid device ordinal`？

**A**: 这是因为 GPU 数量不够。如果你只有 1 个 GPU（如 RTX 4090），使用 `--tp 1 --ep 1` 而不是 `--tp 2 --ep 2`。

### Q: Docker 无法使用 GPU？

**A**: 如果你之前成功运行过，应该没问题。如果遇到问题，检查：
1. Docker Desktop 是否运行
2. NVIDIA Container Toolkit 是否安装
3. Docker Desktop 设置中是否启用 GPU 支持

### Q: 模型下载很慢？

**A**: 
```powershell
# 设置镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### Q: 代码修改不生效？

**A**: 确保在 Docker 容器内重新安装: `pip install -e "python[all]"`

---

## 需要帮助？

查看详细文档:
- **[WORKFLOW.md](./WORKFLOW.md)** - 完整流程（推荐）
- **[DOCKER_TESTING_GUIDE.md](./DOCKER_TESTING_GUIDE.md)** - Docker 详细说明
- **[test/README_DEBUG.md](./test/README_DEBUG.md)** - 测试脚本说明
