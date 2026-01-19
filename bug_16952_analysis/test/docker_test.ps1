# Docker 测试脚本 - Bug #16952
# 使用官方 Docker 镜像进行测试

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bug #16952 Docker 测试" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 配置
$IMAGE = "lmsysorg/sglang:latest"
$CONTAINER_NAME = "sglang_bug16952_test"
$MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
$PORT = 32123
$HF_CACHE = "$env:USERPROFILE\.cache\huggingface"

# 检查 Docker
Write-Host "[检查] 检查 Docker 是否运行..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] Docker 未安装或未运行" -ForegroundColor Red
        Write-Host "请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[检查] OK $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[错误] Docker 未安装或未运行" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 检查 NVIDIA Container Toolkit
Write-Host "[检查] 检查 GPU 支持..." -ForegroundColor Yellow
try {
    $gpuTest = docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[警告] GPU 支持可能未启用" -ForegroundColor Yellow
        Write-Host "请确保已安装 NVIDIA Container Toolkit" -ForegroundColor Yellow
        Write-Host "Windows: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" -ForegroundColor Gray
    } else {
        Write-Host "[检查] OK GPU 支持已启用" -ForegroundColor Green
    }
} catch {
    Write-Host "[警告] 无法验证 GPU 支持" -ForegroundColor Yellow
}
Write-Host ""

# 检查镜像
Write-Host "[检查] 检查 Docker 镜像..." -ForegroundColor Yellow
$imageExists = docker images $IMAGE --format "{{.Repository}}:{{.Tag}}" 2>&1
if ($imageExists -notmatch $IMAGE) {
    Write-Host "[下载] 拉取 Docker 镜像: $IMAGE" -ForegroundColor Yellow
    Write-Host "（这可能需要几分钟）" -ForegroundColor Gray
    docker pull $IMAGE
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 镜像拉取失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "[下载] OK 镜像拉取完成" -ForegroundColor Green
} else {
    Write-Host "[检查] OK 镜像已存在: $IMAGE" -ForegroundColor Green
}
Write-Host ""

# 停止并删除旧容器（如果存在）
Write-Host "[清理] 清理旧容器..." -ForegroundColor Yellow
docker stop $CONTAINER_NAME 2>&1 | Out-Null
docker rm $CONTAINER_NAME 2>&1 | Out-Null
Write-Host "[清理] OK 清理完成" -ForegroundColor Green
Write-Host ""

# 检查模型缓存
Write-Host "[检查] 检查模型缓存..." -ForegroundColor Yellow
if (Test-Path $HF_CACHE) {
    $cacheSize = (Get-ChildItem $HF_CACHE -Recurse -File -ErrorAction SilentlyContinue | 
        Measure-Object -Property Length -Sum).Sum
    $cacheSizeGB = [math]::Round($cacheSize / 1GB, 2)
    Write-Host "[检查] OK 模型缓存: $cacheSizeGB GB" -ForegroundColor Green
} else {
    Write-Host "[检查] 模型缓存目录不存在，将创建" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $HF_CACHE -Force | Out-Null
}
Write-Host ""

# 构建 Docker 命令
Write-Host "[配置] 构建 Docker 命令..." -ForegroundColor Yellow

# Docker 命令参数
$dockerArgs = @(
    "run",
    "--rm",
    "--gpus", "all",
    "--shm-size", "32g",
    "-p", "${PORT}:${PORT}",
    "-v", "${HF_CACHE}:/root/.cache/huggingface",
    "--ipc=host"
)

# Windows 上 network=host 可能不工作，但保留也不会有害
# 如果遇到问题，可以注释掉下面两行
$dockerArgs += "--network=host"
$dockerArgs += "--privileged"

$dockerArgs += $IMAGE
$dockerArgs += "python3", "-m", "sglang.launch_server"
$dockerArgs += "--model-path", $MODEL_PATH
$dockerArgs += "--tp", "2"
$dockerArgs += "--ep", "2"
$dockerArgs += "--port", $PORT.ToString()
$dockerArgs += "--trust-remote-code"
$dockerArgs += "--enable-dp-attention"
$dockerArgs += "--disable-custom-all-reduce"
$dockerArgs += "--enable-two-batch-overlap"
$dockerArgs += "--moe-a2a-backend", "deepep"
$dockerArgs += "--log-level", "info"

Write-Host "[配置] OK 命令已构建" -ForegroundColor Green
Write-Host ""

# 显示命令
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "运行测试命令" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Docker 命令:" -ForegroundColor Yellow
Write-Host "docker $($dockerArgs -join ' ')" -ForegroundColor Gray
Write-Host ""
Write-Host "提示:" -ForegroundColor Yellow
Write-Host "  - 首次运行会下载模型（可能需要 10-30 分钟）" -ForegroundColor Gray
Write-Host "  - 按 Ctrl+C 可以中断" -ForegroundColor Gray
Write-Host "  - 查看日志以确认修复是否生效" -ForegroundColor Gray
Write-Host ""
Write-Host "预期日志（修复后）:" -ForegroundColor Yellow
Write-Host "  TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl..." -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始运行..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 运行 Docker 命令
try {
    & docker $dockerArgs
} catch {
    Write-Host ""
    Write-Host "[错误] Docker 运行出错: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "故障排除:" -ForegroundColor Yellow
    Write-Host "  1. 确保 Docker Desktop 正在运行" -ForegroundColor Gray
    Write-Host "  2. 确保已安装 NVIDIA Container Toolkit" -ForegroundColor Gray
    Write-Host "  3. 检查 GPU 是否可用: docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "测试完成！" -ForegroundColor Green
