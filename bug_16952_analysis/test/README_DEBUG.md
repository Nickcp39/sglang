# 调试和监控脚本说明

## ⚠️ 重要：强烈建议使用 Docker

**对于 Bug #16952，强烈建议使用 Docker 进行测试**，原因：
1. DeepEP 安装复杂，Docker 镜像已预装
2. NCCL 版本正确，避免通信错误
3. 复现一致性，与用户环境一致

**推荐使用 `docker_test.ps1`**（见下方）

详见：[../DOCKER_TESTING_GUIDE.md](../DOCKER_TESTING_GUIDE.md)

---

## 脚本列表

### 0. `docker_test.ps1` - Docker 测试（⭐ 强烈推荐）

**功能**：使用官方 Docker 镜像进行测试

**使用**：
```powershell
cd bug_16952_analysis\test
.\docker_test.ps1
```

**特点**：
- ✅ 使用官方镜像 `lmsysorg/sglang:latest`
- ✅ DeepEP 已预装
- ✅ NCCL 版本正确
- ✅ 自动挂载模型缓存
- ✅ 复现一致性最好

### 1. `simple_test.ps1` - 简单测试（本地环境）

**功能**：直接运行测试，实时显示所有输出

**使用**：
```powershell
cd bug_16952_analysis\test
.\simple_test.ps1
```

**特点**：
- ✅ 简单直接
- ✅ 实时显示所有输出
- ✅ 可以看到下载和启动过程

### 2. `run_with_monitoring.ps1` - 带监控的测试

**功能**：后台监控下载进度和进程状态，同时运行测试

**使用**：
```powershell
cd bug_16952_analysis\test
.\run_with_monitoring.ps1
```

**特点**：
- ✅ 实时显示下载进度（每10秒更新）
- ✅ 实时显示进程状态（每15秒更新）
- ✅ 显示关键日志信息
- ✅ 保存完整日志到文件

**输出**：
- `test_output.log` - 标准输出
- `test_error.log` - 错误输出
- 实时监控信息显示在屏幕上

### 3. `monitor_download.ps1` - 仅监控下载

**功能**：只监控模型下载进度（在另一个终端运行）

**使用**：
```powershell
cd bug_16952_analysis\test
.\monitor_download.ps1
```

**特点**：
- ✅ 每10秒显示下载进度
- ✅ 显示下载速度
- ✅ 检测下载是否停止

### 4. `check_process.ps1` - 检查进程状态

**功能**：检查 Python 进程、端口、GPU 状态

**使用**：
```powershell
cd bug_16952_analysis\test
.\check_process.ps1
```

## 推荐使用流程

### 方式 1：Docker 测试（⭐ 强烈推荐）

```powershell
cd bug_16952_analysis\test
.\docker_test.ps1
```

这会：
- 使用官方 Docker 镜像
- 自动检查 GPU 支持
- 自动拉取镜像（如果需要）
- 挂载本地模型缓存
- 运行测试并显示所有输出

### 方式 2：简单测试（本地环境，不推荐）

```powershell
cd bug_16952_analysis\test
.\simple_test.ps1
```

这会：
- 实时显示所有输出
- 可以看到模型下载过程
- 可以看到服务器启动过程
- 可以看到修复是否生效（自动选择日志）

### 方式 2：详细监控

**终端 1**：运行测试
```powershell
cd bug_16952_analysis\test
.\run_with_monitoring.ps1
```

**终端 2**（可选）：监控下载
```powershell
cd bug_16952_analysis\test
.\monitor_download.ps1
```

## 如何判断测试进度

### 1. 模型下载阶段

**标志**：
- 看到 "Downloading..." 相关日志
- 或者使用 `monitor_download.ps1` 看到文件大小增长

**时间**：可能需要 10-30 分钟（取决于网络）

### 2. 模型加载阶段

**标志**：
- 看到 "Loading model..." 相关日志
- GPU 显存开始增加（可以用 `nvidia-smi` 查看）

**时间**：可能需要几分钟

### 3. 服务器启动阶段

**标志**：
- 看到 "Server started" 或类似日志
- 端口 32123 开始监听
- 看到自动选择日志：`"TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl..."`

**时间**：可能需要几分钟

### 4. 测试成功标志

**关键日志**：
```
TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.
```

**其他标志**：
- 服务器成功启动
- 没有错误信息
- 端口 32123 在监听

## 常见问题

### Q: 看不到任何输出？

**A**: 
1. 检查进程是否在运行：`Get-Process python*`
2. 检查日志文件：`test_output.log` 和 `test_error.log`
3. 使用 `simple_test.ps1` 直接运行

### Q: 下载很慢？

**A**: 
1. 使用镜像：`$env:HF_ENDPOINT = "https://hf-mirror.com"`
2. 检查网络连接
3. 使用 `monitor_download.ps1` 查看实际下载速度

### Q: 如何知道修复是否生效？

**A**: 
查找日志中的这一行：
```
TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.
```

如果看到这行，说明自动选择功能工作正常！

## 快速测试命令

### 测试 1：自动选择（修复后应该成功）

```powershell
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep
```

**预期**：
- ✅ 应该自动选择 `flashinfer_cutedsl` backend
- ✅ 应该能看到日志：`"TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend..."`
- ✅ 应该能正常启动

### 测试 2：显式指定（应该成功）

```powershell
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --moe-runner-backend flashinfer_cutedsl
```

**预期**：
- ✅ 应该能正常启动

### 测试 3：错误配置（应该失败并显示清晰错误）

```powershell
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --moe-runner-backend triton
```

**预期**：
- ❌ 应该失败
- ✅ 错误信息：`"TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. Please use --moe-runner-backend flashinfer_cutedsl."`
