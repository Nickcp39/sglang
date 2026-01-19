# 模型管理指南

## 📍 模型存储位置

所有 HuggingFace 模型都下载到：

**Windows**:
```
C:\Users\<你的用户名>\.cache\huggingface\hub\
```

**Linux/Mac**:
```
~/.cache/huggingface/hub/
```

## 🔍 查看已下载的模型

### 方法 1: 使用 PowerShell 查看（推荐）

```powershell
# 查看缓存目录
$cacheDir = "$env:USERPROFILE\.cache\huggingface\hub"
Get-ChildItem $cacheDir -Directory | Select-Object Name, @{Name="Size(GB)";Expression={[math]::Round((Get-ChildItem $_.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB, 2)}}
```

### 方法 2: 使用文件资源管理器

1. 打开文件资源管理器
2. 输入路径：`%USERPROFILE%\.cache\huggingface\hub`
3. 查看各个模型文件夹的大小

### 方法 3: 使用 huggingface-cli（如果已安装）

```powershell
# 安装（如果还没有）
pip install huggingface_hub

# 查看缓存信息
huggingface-cli scan-cache
```

这会显示：
- 所有已下载的模型
- 每个模型的大小
- 最后使用时间
- 可以安全删除的模型

### 方法 4: 快速查看（PowerShell 一行命令）

```powershell
# 查看所有模型及其大小（按大小排序）
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Directory | ForEach-Object { [PSCustomObject]@{ Model = $_.Name; SizeGB = [math]::Round((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB, 2) } } | Sort-Object SizeGB -Descending | Format-Table -AutoSize
```

## 📊 查看特定模型的大小

```powershell
# 查看 Qwen1.5-MoE-A2.7B-Chat 的大小
$modelPath = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
if (Test-Path $modelPath) {
    $size = (Get-ChildItem $modelPath -Recurse -File | Measure-Object -Property Length -Sum).Sum
    Write-Host "模型大小: $([math]::Round($size / 1GB, 2)) GB"
} else {
    Write-Host "模型尚未下载"
}
```

## 🗑️ 删除不需要的模型

### ⚠️ 重要：模型不会自动删除

HuggingFace 缓存**不会自动删除**，需要手动清理。

### 方法 1: 使用 PowerShell 删除特定模型

```powershell
# 删除 Qwen1.5-MoE-A2.7B-Chat（如果不需要了）
$modelPath = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
if (Test-Path $modelPath) {
    Remove-Item $modelPath -Recurse -Force
    Write-Host "已删除模型"
} else {
    Write-Host "模型不存在"
}
```

### 方法 2: 使用 huggingface-cli 删除（推荐）

```powershell
# 安装（如果还没有）
pip install huggingface_hub

# 查看可以删除的模型
huggingface-cli scan-cache

# 删除特定模型
huggingface-cli delete-cache --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat
```

### 方法 3: 清理所有未使用的模型

```powershell
# 使用 huggingface-cli 清理（只删除未使用的）
huggingface-cli scan-cache --dir "$env:USERPROFILE\.cache\huggingface"
# 然后根据提示删除
```

### 方法 4: 手动删除（文件资源管理器）

1. 打开 `%USERPROFILE%\.cache\huggingface\hub`
2. 找到对应的模型文件夹（格式：`models--<org>--<model-name>`）
3. 右键删除整个文件夹

## 💾 常见模型大小参考

| 模型 | 大小（GB） |
|------|-----------|
| Qwen/Qwen1.5-MoE-A2.7B-Chat | ~28-30 GB |
| inclusionAI/Ling-lite | ~33.6 GB |
| microsoft/Phi-3.5-MoE-instruct | ~83.8 GB |
| deepseek-ai/DeepSeek-V3 | ~数百 GB |

## 📝 检查磁盘空间

```powershell
# 查看整个 HuggingFace 缓存目录的大小
$cacheDir = "$env:USERPROFILE\.cache\huggingface"
$totalSize = (Get-ChildItem $cacheDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
Write-Host "HuggingFace 缓存总大小: $([math]::Round($totalSize / 1GB, 2)) GB"

# 查看磁盘剩余空间
Get-PSDrive C | Select-Object Used, Free, @{Name="Free(GB)";Expression={[math]::Round($_.Free / 1GB, 2)}}
```

## 🔄 模型缓存的工作原理

1. **首次下载**：模型会完整下载到缓存目录
2. **后续使用**：直接从缓存读取，无需重新下载
3. **不会自动删除**：即使不再使用，模型也会保留在缓存中
4. **共享使用**：所有使用 HuggingFace 库的程序共享同一个缓存目录

## ⚡ 快速命令参考

```powershell
# 查看所有模型（一行命令）
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Directory | Select-Object Name

# 查看缓存总大小
$size = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum; Write-Host "$([math]::Round($size / 1GB, 2)) GB"

# 删除特定模型（替换 MODEL_NAME）
Remove-Item "$env:USERPROFILE\.cache\huggingface\hub\models--MODEL_NAME" -Recurse -Force
```

## 🎯 建议

1. **定期清理**：如果磁盘空间紧张，定期检查并删除不需要的模型
2. **使用 huggingface-cli**：这是最安全和推荐的方式
3. **备份重要模型**：如果需要保留某些模型，可以复制到其他位置
4. **监控磁盘空间**：大型模型会占用大量空间，注意磁盘使用情况
