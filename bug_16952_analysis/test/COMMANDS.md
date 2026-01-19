# Bug #16952 测试命令（直接复制运行）

## 步骤 1: 检查 Docker 和 GPU（可选）

```powershell
# 检查 Docker
docker --version

# 检查 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## 步骤 2: 拉取镜像（如果还没有）

```powershell
docker pull lmsysorg/sglang:latest
```

**等待**: 可能需要 5-15 分钟

---

## 步骤 2.5: 清理之前的容器（重要！）

**在运行测试前，先清理不需要的容器，避免 GPU 内存被占用**：

```powershell
# 停止所有 sglang 测试容器
docker stop sglang_bug16952 intelligent_wiles clever_wozniak 2>$null

# 删除这些容器
docker rm sglang_bug16952 intelligent_wiles clever_wozniak 2>$null

# 检查 GPU 内存（可选）
nvidia-smi
```

**注意**: 如果你有其他正在使用的容器（如 `sglang_qwen`），不要关闭它。

---

## 步骤 3: 测试修复前（复现 Bug）

**重要**: 如果你只有 1 个 GPU（如 RTX 4090），使用 `--tp 1 --ep 1`。如果有 2 个或更多 GPU，可以使用 `--tp 2 --ep 2`。

### 方式 A: 单 GPU（RTX 4090 等）

**注意**: RTX 4090 不支持 `deep_gemm`（需要 SM >= 90），所以必须使用 `--deepep-mode normal` 来绕过 deep_gemm 要求。

**内存优化**: RTX 4090（24 GB）运行 Qwen1.5-MoE-A2.7B-Chat（~22 GB 模型）显存紧张，已添加内存优化参数。

```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --deepep-mode normal --mem-fraction-static 0.75 --chunked-prefill-size 2048 --max-running-requests 32 --log-level info
```

### 方式 B: 多 GPU（2 个或更多，SM >= 90）

**注意**: 如果 GPU 支持 SM >= 90（如 H100, H20），可以使用默认的 `--deepep-mode auto`。

```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

### 方式 C: 多 GPU（SM < 90，如 RTX 4090 x2）

```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --deepep-mode normal --log-level info
```

### 方式 D: 多行命令（更易读，单 GPU）

```powershell
docker run --rm --gpus all --shm-size 32g `
    -p 32123:32123 `
    -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
    --ipc=host --network=host --privileged `
    lmsysorg/sglang:latest `
    python3 -m sglang.launch_server `
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat `
        --tp 1 --ep 1 --port 32123 `
        --trust-remote-code `
        --enable-dp-attention `
        --disable-custom-all-reduce `
        --enable-two-batch-overlap `
        --moe-a2a-backend deepep `
        --deepep-mode normal `
        --mem-fraction-static 0.75 `
        --chunked-prefill-size 2048 `
        --max-running-requests 32 `
        --log-level info
```

**预期结果**: 应该看到错误 `AssertionError: forward_deepgemm_masked is deprecated`

**按 Ctrl+C 中断**

---

## 步骤 4: 应用修复

```powershell
# 进入项目根目录
cd d:\Yc-software\github_code\sglang

# 应用修复
git apply bug_16952_analysis\code\patch_server_args.py.patch
git apply bug_16952_analysis\code\patch_ep_moe_layer.py.patch

# 验证修复已应用
Select-String -Path python\sglang\srt\server_args.py -Pattern "TBO/SBO is enabled with DeepEP"
Select-String -Path python\sglang\srt\layers\moe\ep_moe\layer.py -Pattern "TBO/SBO requires --moe-runner-backend"
```

**预期输出**: 应该找到匹配的行

---

## 步骤 5: 测试修复后（验证修复）

### 5.1 启动容器并挂载代码

```powershell
# 停止之前的容器（如果有）
docker stop sglang_bug16952 2>$null
docker rm sglang_bug16952 2>$null

# 启动容器并挂载代码
docker run -itd --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" -v "${PWD}:/sgl-workspace/sglang" --ipc=host --network=host --privileged --name sglang_bug16952 lmsysorg/sglang:latest /bin/bash
```

### 5.2 在容器内重新安装 sglang

```powershell
# 进入容器
docker exec -it sglang_bug16952 /bin/bash

# 在容器内执行（复制这些命令到容器内）
cd /sgl-workspace/sglang
pip install -e "python[all]"

# 退出容器
exit
```

### 5.3 运行测试（修复后）

**单 GPU**（RTX 4090 等，需要 `--deepep-mode normal`）:
```powershell
docker exec -it sglang_bug16952 python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --deepep-mode normal --mem-fraction-static 0.75 --chunked-prefill-size 2048 --max-running-requests 32 --log-level info
```

**多 GPU**:
```powershell
docker exec -it sglang_bug16952 python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

**预期结果**: 
- 应该看到日志: `TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.`
- 然后服务器成功启动

**按 Ctrl+C 中断**

---

## 步骤 6: 清理

```powershell
# 停止并删除容器
docker stop sglang_bug16952
docker rm sglang_bug16952

# 如果需要，恢复原始代码
git checkout python/sglang/srt/server_args.py
git checkout python/sglang/srt/layers/moe/ep_moe/layer.py
```

---

## 快速参考

### 修复前测试（一行命令，单 GPU，RTX 4090）

```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --deepep-mode normal --mem-fraction-static 0.75 --chunked-prefill-size 2048 --max-running-requests 32 --log-level info
```

### 应用修复（两行命令）

```powershell
cd d:\Yc-software\github_code\sglang
git apply bug_16952_analysis\code\patch_server_args.py.patch; git apply bug_16952_analysis\code\patch_ep_moe_layer.py.patch
```

### 测试修复后（需要先启动容器）

```powershell
# 启动容器（一次）
docker run -itd --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" -v "${PWD}:/sgl-workspace/sglang" --ipc=host --network=host --privileged --name sglang_bug16952 lmsysorg/sglang:latest /bin/bash

# 重新安装（一次）
docker exec sglang_bug16952 bash -c "cd /sgl-workspace/sglang && pip install -e 'python[all]'"

# 运行测试（可以多次运行，单 GPU，RTX 4090）
docker exec -it sglang_bug16952 python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --deepep-mode normal --mem-fraction-static 0.75 --chunked-prefill-size 2048 --max-running-requests 32 --log-level info
```

---

## 注意事项

0. **清理容器**: 在运行测试前，确保关闭不需要的容器，避免 GPU 内存被占用导致 Docker 变慢
1. **GPU 数量**: 
   - **单 GPU**（如 RTX 4090）: 使用 `--tp 1 --ep 1`
   - **多 GPU**（2 个或更多）: 使用 `--tp 2 --ep 2` 或更多
   - 如果看到 `CUDA error: invalid device ordinal`，说明 GPU 数量不够，减少 `--tp` 和 `--ep` 的值
2. **DeepEP Mode**:
   - **RTX 4090 等 SM < 90 的 GPU**: 必须使用 `--deepep-mode normal`（因为不支持 deep_gemm）
   - **H100/H20 等 SM >= 90 的 GPU**: 可以使用默认的 `--deepep-mode auto` 或不指定
   - 如果看到 `AssertionError: DeepEP DeepEPMode.AUTO mode requires deep_gemm`，添加 `--deepep-mode normal`
3. **内存不足（RTX 4090）**:
   - 如果看到 `RuntimeError: Not enough memory`，说明显存不足
   - 已添加 `--mem-fraction-static 0.75 --chunked-prefill-size 2048 --max-running-requests 32` 来减少内存使用
   - 如果仍然不足，可以进一步减少：`--mem-fraction-static 0.7` 或 `--chunked-prefill-size 1024`
2. **端口**: 使用 port 32123（不会与你 port 30000 的东西冲突）
3. **模型下载**: 首次运行会自动下载模型（可能需要 10-30 分钟）
4. **中断**: 按 Ctrl+C 可以中断运行
5. **容器**: 测试修复后需要挂载代码，所以需要先启动容器
