# 模型下载说明

## ⚠️ 重要：对于 Bug #16952，强烈建议使用 Docker

虽然模型下载本身不需要 Docker，但**为了复现和测试 Bug #16952，强烈建议使用 Docker**，原因：

1. **DeepEP 依赖复杂**：DeepEP 安装过程非常繁琐，官方 Docker 镜像已预装
2. **NCCL 版本正确**：TBO 需要正确的 NCCL 版本，Docker 镜像已配置
3. **复现一致性**：用户在高性能计算集群上，使用官方镜像能保证一致性

详见：[DOCKER_TESTING_GUIDE.md](./DOCKER_TESTING_GUIDE.md)

## 模型下载本身不需要 Docker

模型下载是通过 **HuggingFace 库**完成的，在本地 Python 环境中就可以工作。

### 手动预下载模型（可选）

如果你想先下载模型再测试：

```powershell
# 使用 Python 下载模型
python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B-Chat'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B-Chat')"
```

或者使用 huggingface-cli：

```powershell
# 安装 huggingface-cli（如果还没有）
pip install huggingface_hub

# 下载模型
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat
```

## 模型下载的工作原理

### 1. 自动下载

当你运行：

```powershell
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat ...
```

sglang 内部会：
1. 使用 `transformers` 库或 `huggingface_hub` 库
2. 从 HuggingFace Hub 下载模型
3. 保存到本地缓存：`C:\Users\<你的用户名>\.cache\huggingface\`

### 2. 下载位置

模型会下载到：
```
C:\Users\<你的用户名>\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat\
```

### 3. 下载过程

- **首次运行**：自动下载（可能需要 10-30 分钟）
- **后续运行**：直接使用缓存，不需要重新下载
- **下载时可能没有输出**：这是正常的，模型在后台下载

## 本地 Python 环境 vs Docker

### 本地 Python 环境（推荐）

**优点**：
- ✅ 简单直接，不需要 Docker
- ✅ 模型下载到本地，可以重复使用
- ✅ 调试更方便

**使用方式**：
```powershell
# 直接运行，模型会自动下载
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 ...
```

### Docker（可选）

**使用场景**：
- 想要隔离的环境
- 想要使用预配置的镜像
- 多平台部署

**使用方式**：
```powershell
# Docker 也会自动下载模型（到容器内的缓存）
docker run --gpus all -it --rm `
  -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
  lmsysorg/sglang:latest `
  python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat ...
```

**注意**：即使使用 Docker，模型也会下载到挂载的本地目录（通过 `-v` 参数）

## 验证模型下载

### 检查模型是否在下载

```powershell
# 监控下载进度
$cachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
while ($true) {
    if (Test-Path $cachePath) {
        $files = Get-ChildItem $cachePath -Recurse -File -ErrorAction SilentlyContinue
        if ($files) {
            $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
            $sizeGB = [math]::Round($totalSize / 1GB, 2)
            $fileCount = $files.Count
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 已下载: $sizeGB GB ($fileCount 个文件)"
        }
    } else {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 等待开始下载..."
    }
    Start-Sleep -Seconds 10
}
```

### 检查模型是否已下载

```powershell
.\bug_16952_analysis\check_model_and_test.ps1 -CheckOnly
```

## 推荐方案

### 方案 1: 使用 Docker（强烈推荐用于 Bug #16952）

```powershell
# 使用 Docker 测试脚本
cd bug_16952_analysis\test
.\docker_test.ps1
```

**为什么推荐 Docker**：
1. ✅ DeepEP 已预装，避免复杂的安装过程
2. ✅ NCCL 版本正确，避免通信错误
3. ✅ 复现一致性，与用户环境一致
4. ✅ 节省时间，避免环境配置问题

### 方案 2: 本地 Python 环境（仅用于简单测试）

```powershell
# 直接运行，模型会自动下载
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep
```

**注意**：
- 需要手动安装 DeepEP（可能很复杂）
- 需要确保 NCCL 版本正确
- 可能遇到环境相关的问题，难以区分是 Bug 还是环境问题

## 如果下载很慢

可以使用镜像站点：

```powershell
# 设置 HuggingFace 镜像（如果在中国）
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 然后再运行命令
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat ...
```

## 总结

### 对于 Bug #16952 测试：
- ✅ **强烈建议使用 Docker**（DeepEP、NCCL 已预装）
- ✅ **模型会自动下载到挂载的本地缓存**
- ✅ **使用 `docker_test.ps1` 脚本**

### 模型下载本身：
- ✅ **模型会自动下载到本地缓存**（无论是否使用 Docker）
- ✅ **Docker 中通过 `-v` 挂载本地缓存，避免重复下载**
