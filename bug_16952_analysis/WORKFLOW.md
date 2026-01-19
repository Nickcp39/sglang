# Bug #16952 完整测试流程

## 概述

本文档提供从 Docker 环境准备到测试完成的完整流程，每一步都有详细说明。

## 🚀 快速开始（Docker 已安装）

如果你已经安装好 Docker 并且之前成功运行过容器，可以直接：

```powershell
# 1. 快速检查 GPU（可选，如果之前运行过应该没问题）
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# 2. 拉取镜像（如果还没有）
docker pull lmsysorg/sglang:latest

# 3. 直接测试修复前（复现 Bug）
# 如果你在项目根目录，使用：
cd bug_16952_analysis\test
.\docker_test.ps1

# 如果你已经在 bug_16952_analysis 目录，使用：
cd test
.\docker_test.ps1
```

然后按照步骤 5-7 继续。

---

## 流程概览

```
1. 环境准备（Docker）← 如果你已安装，可以跳过
   ↓
2. 拉取镜像
   ↓
3. 准备代码修复
   ↓
4. 测试修复前（复现 Bug）
   ↓
5. 应用修复
   ↓
6. 测试修复后（验证修复）
   ↓
7. 记录结果
```

---

## 步骤 1: 环境准备（Docker）

### 1.1 检查 Docker 是否安装

**Windows PowerShell**:
```powershell
docker --version
```

**预期输出**:
```
Docker version 24.0.0, build ...
```

**如果未安装**:
- 下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
- 重启电脑

### 1.2 检查 Docker 是否运行

```powershell
docker ps
```

**预期输出**: 显示容器列表（可能为空，但不应报错）

**如果报错**:
- 启动 Docker Desktop
- 等待 Docker 完全启动（系统托盘图标不再闪烁）

### 1.3 检查 GPU 支持

```powershell
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**预期输出**: 显示 GPU 信息（类似 `nvidia-smi` 的输出）

**如果报错**:
1. 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. 在 Docker Desktop 设置中启用 GPU 支持
3. 重启 Docker Desktop

**注意**: 如果你之前已经在 port 30000 成功运行过其他容器，说明 GPU 支持已经配置好了，可以跳过这个检查。

### 1.4 检查模型缓存目录

```powershell
# 检查缓存目录是否存在
Test-Path "$env:USERPROFILE\.cache\huggingface"

# 如果不存在，创建它
New-Item -ItemType Directory -Path "$env:USERPROFILE\.cache\huggingface" -Force
```

**说明**: 模型会下载到这个目录，Docker 会挂载它以避免重复下载。

---

## 步骤 2: 拉取 Docker 镜像

### 2.1 拉取官方镜像

```powershell
docker pull lmsysorg/sglang:latest
```

**预期输出**: 
```
latest: Pulling from lmsysorg/sglang
...
Status: Downloaded newer image for lmsysorg/sglang:latest
```

**时间**: 可能需要 5-15 分钟（取决于网络速度）

**说明**: 
- 镜像大小约 10-20 GB
- 包含 DeepEP、NCCL、FlashInfer 等所有依赖
- 这是官方维护的镜像，确保环境一致性

### 2.2 验证镜像

```powershell
docker images lmsysorg/sglang:latest
```

**预期输出**: 显示镜像信息

---

## 步骤 3: 准备代码修复

### 3.1 检查当前代码状态

```powershell
# 进入项目目录
cd d:\Yc-software\github_code\sglang

# 检查是否在正确的分支
git branch

# 查看修改状态
git status
```

**说明**: 
- 确保在正确的分支（建议创建新分支）
- 如果代码已经修改，先备份或提交

### 3.2 查看修复代码

```powershell
# 查看修复说明
cat bug_16952_analysis\code\README.md

# 查看修复内容
cat bug_16952_analysis\code\fix_server_args.py
cat bug_16952_analysis\code\fix_ep_moe_layer.py
```

**说明**: 
- 修复代码已经准备好
- 两个文件需要修改：
  - `python/sglang/srt/server_args.py`
  - `python/sglang/srt/layers/moe/ep_moe/layer.py`

---

## 步骤 4: 测试修复前（复现 Bug）

### 4.1 运行测试（修复前）

**推荐：直接运行命令**（最简单）:

**单 GPU**（如 RTX 4090）:
```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

**多 GPU**（2 个或更多）:
```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" --ipc=host --network=host --privileged lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info
```

**注意**: 如果看到 `CUDA error: invalid device ordinal`，说明 GPU 数量不够，使用 `--tp 1 --ep 1`

**或者使用脚本**（如果脚本能运行）:
```powershell
cd bug_16952_analysis\test
.\docker_test.ps1
```

**或者手动运行**（多行格式，更易读）:

```powershell
docker run --rm --gpus all --shm-size 32g -p 32123:32123 `
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
        --log-level info
```

### 4.2 预期结果（修复前）

**应该看到错误**:
```
AssertionError: forward_deepgemm_masked is deprecated
```

**或者**:
```
RuntimeError: TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. Please use --moe-runner-backend flashinfer_cutedsl.
```

**说明**: 
- 这是预期的错误，证明 Bug 存在
- 如果看到这个错误，说明环境配置正确

### 4.3 记录结果

```powershell
# 创建结果目录
New-Item -ItemType Directory -Path "bug_16952_analysis\result\before_fix" -Force

# 保存日志（如果脚本有输出日志）
# 或者手动记录错误信息
```

**记录内容**:
- 错误类型
- 错误信息
- 错误堆栈（如果有）
- 运行时间
- 模型下载状态

---

## 步骤 5: 应用修复

### 5.1 备份原始文件

```powershell
# 备份 server_args.py
Copy-Item python\sglang\srt\server_args.py python\sglang\srt\server_args.py.backup

# 备份 layer.py
Copy-Item python\sglang\srt\layers\moe\ep_moe\layer.py python\sglang\srt\layers\moe\ep_moe\layer.py.backup
```

### 5.2 应用修复

**方法 1: 使用脚本（推荐）**

```powershell
cd bug_16952_analysis\test
python apply_fix.py
```

**方法 2: 手动应用**

查看 `bug_16952_analysis\code\fix_server_args.py` 和 `fix_ep_moe_layer.py`，手动修改代码。

**方法 3: 使用 Git Patch**

```powershell
# 应用 patch
git apply bug_16952_analysis\code\patch_server_args.py.patch
git apply bug_16952_analysis\code\patch_ep_moe_layer.py.patch
```

### 5.3 验证修复已应用

```powershell
# 检查 server_args.py 是否包含修复代码
Select-String -Path python\sglang\srt\server_args.py -Pattern "TBO/SBO is enabled with DeepEP"

# 检查 layer.py 是否包含修复代码
Select-String -Path python\sglang\srt\layers\moe\ep_moe\layer.py -Pattern "TBO/SBO requires --moe-runner-backend"
```

**预期输出**: 应该找到匹配的行

---

## 步骤 6: 测试修复后（验证修复）

### 6.1 在 Docker 中测试修复后的代码

**重要**: 修复后的代码需要在 Docker 中测试，有两种方式：

#### 方式 A: 挂载代码目录（推荐）

```powershell
# 停止之前的容器（如果有）
docker stop sglang_bug16952 2>$null
docker rm sglang_bug16952 2>$null

# 启动容器并挂载代码
docker run -itd --gpus all --shm-size 32g -p 32123:32123 `
    -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
    -v "${PWD}:/sgl-workspace/sglang" `
    --ipc=host --network=host --privileged `
    --name sglang_bug16952 `
    lmsysorg/sglang:latest /bin/bash

# 进入容器
docker exec -it sglang_bug16952 /bin/bash

# 在容器内重新安装 sglang（使修改生效）
cd /sgl-workspace/sglang
pip install -e "python[all]"

# 退出容器
exit
```

#### 方式 B: 使用测试脚本

```powershell
cd bug_16952_analysis\test
.\docker_test_with_fix.ps1
```

### 6.2 运行测试（修复后）

```powershell
# 在容器内运行（单 GPU）
docker exec -it sglang_bug16952 python3 -m sglang.launch_server `
    --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat `
    --tp 1 --ep 1 --port 32123 `
    --trust-remote-code `
    --enable-dp-attention `
    --disable-custom-all-reduce `
    --enable-two-batch-overlap `
    --moe-a2a-backend deepep `
    --log-level info
```

### 6.3 预期结果（修复后）

**应该看到**:
```
TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.
```

**然后**:
- 模型开始加载
- 服务器成功启动
- 没有错误信息

**验证服务器运行**:
```powershell
# 在另一个终端检查端口
netstat -an | findstr 32123

# 或者发送测试请求
curl http://localhost:32123/health
```

### 6.4 记录结果

```powershell
# 创建结果目录
New-Item -ItemType Directory -Path "bug_16952_analysis\result\after_fix" -Force

# 保存日志
# 记录：
# - 是否看到自动选择日志
# - 服务器是否成功启动
# - 是否有错误
# - 运行时间
```

---

## 步骤 7: 记录结果和总结

### 7.1 填写结果摘要

编辑 `bug_16952_analysis\result\SUMMARY.md`:

```markdown
# Bug #16952 测试结果

## 测试环境
- Docker 镜像: lmsysorg/sglang:latest
- 测试模型: Qwen/Qwen1.5-MoE-A2.7B-Chat
- GPU: RTX 4090
- 测试时间: [日期时间]

## 修复前测试
- [ ] 成功复现 Bug
- 错误信息: [记录错误]
- 运行时间: [时间]

## 修复后测试
- [ ] 看到自动选择日志
- [ ] 服务器成功启动
- [ ] 无错误信息
- 运行时间: [时间]

## 结论
[填写结论]
```

### 7.2 清理

```powershell
# 停止并删除测试容器
docker stop sglang_bug16952
docker rm sglang_bug16952

# 如果需要，恢复原始代码
Copy-Item python\sglang\srt\server_args.py.backup python\sglang\srt\server_args.py
Copy-Item python\sglang\srt\layers\moe\ep_moe\layer.py.backup python\sglang\srt\layers\moe\ep_moe\layer.py
```

---

## 故障排除

### 问题 1: Docker 无法使用 GPU

**解决**:
1. 检查 NVIDIA Container Toolkit 是否安装
2. 检查 Docker Desktop 设置中的 GPU 支持
3. 重启 Docker Desktop

### 问题 2: 模型下载很慢

**解决**:
```powershell
# 设置镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 或者在容器内
docker exec -it sglang_bug16952 bash -c "export HF_ENDPOINT=https://hf-mirror.com && python3 -m sglang.launch_server ..."
```

### 问题 3: 代码修改不生效

**解决**:
1. 确保代码已挂载到容器
2. 在容器内重新安装: `pip install -e "python[all]"`
3. 检查 Python 缓存: `find /sgl-workspace/sglang -name "*.pyc" -delete`

### 问题 4: 端口被占用

**解决**:
```powershell
# 检查端口
netstat -an | findstr 32123

# 更改端口
# 在命令中使用 --port 32124
```

---

## 快速参考

### 一键测试脚本

创建 `bug_16952_analysis\test\full_workflow.ps1` 来自动化整个流程。

### 关键命令速查

```powershell
# 拉取镜像
docker pull lmsysorg/sglang:latest

# 运行测试（修复前）
.\test\docker_test.ps1

# 应用修复
python test\apply_fix.py

# 运行测试（修复后）
.\test\docker_test_with_fix.ps1

# 查看结果
cat result\SUMMARY.md
```

---

## 下一步

测试完成后：
1. 提交代码修复
2. 创建 Pull Request
3. 更新 GitHub Issue

详见：[FIX_PLAN.md](./FIX_PLAN.md)
