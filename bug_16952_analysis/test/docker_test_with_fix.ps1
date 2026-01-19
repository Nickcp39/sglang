# Docker 测试脚本 - 修复后（挂载代码）

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bug #16952 Docker 测试（修复后）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 配置
$IMAGE = "lmsysorg/sglang:latest"
$CONTAINER_NAME = "sglang_bug16952_fix"
$MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
$PORT = 32123
$HF_CACHE = "$env:USERPROFILE\.cache\huggingface"
$SGLANG_ROOT = (Get-Location).Path

# 检查修复是否已应用
Write-Host "[检查] 检查修复是否已应用..." -ForegroundColor Yellow
$fixInServerArgs = Select-String -Path "python\sglang\srt\server_args.py" -Pattern "TBO/SBO is enabled with DeepEP" -Quiet
$fixInLayer = Select-String -Path "python\sglang\srt\layers\moe\ep_moe\layer.py" -Pattern "TBO/SBO requires --moe-runner-backend" -Quiet

if (-not $fixInServerArgs -or -not $fixInLayer) {
    Write-Host "[错误] 修复代码未应用！" -ForegroundColor Red
    Write-Host "请先运行: python test\apply_fix.py" -ForegroundColor Yellow
    exit 1
}
Write-Host "[检查] ✅ 修复代码已应用" -ForegroundColor Green
Write-Host ""

# 检查 Docker
Write-Host "[检查] 检查 Docker..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    Write-Host "[检查] ✅ Docker 已安装" -ForegroundColor Green
} catch {
    Write-Host "[错误] Docker 未安装或未运行" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 停止并删除旧容器
Write-Host "[清理] 清理旧容器..." -ForegroundColor Yellow
docker stop $CONTAINER_NAME 2>&1 | Out-Null
docker rm $CONTAINER_NAME 2>&1 | Out-Null
Write-Host "[清理] ✅ 清理完成" -ForegroundColor Green
Write-Host ""

# 启动容器并挂载代码
Write-Host "[启动] 启动 Docker 容器（挂载代码）..." -ForegroundColor Yellow
Write-Host "  镜像: $IMAGE" -ForegroundColor Gray
Write-Host "  容器名: $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "  代码目录: $SGLANG_ROOT" -ForegroundColor Gray
Write-Host ""

docker run -itd --gpus all --shm-size 32g -p ${PORT}:${PORT} `
    -v "${HF_CACHE}:/root/.cache/huggingface" `
    -v "${SGLANG_ROOT}:/sgl-workspace/sglang" `
    --ipc=host --network=host --privileged `
    --name $CONTAINER_NAME `
    $IMAGE /bin/bash

if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] 容器启动失败" -ForegroundColor Red
    exit 1
}
Write-Host "[启动] ✅ 容器已启动" -ForegroundColor Green
Write-Host ""

# 在容器内重新安装 sglang
Write-Host "[安装] 在容器内重新安装 sglang（使修复生效）..." -ForegroundColor Yellow
Write-Host "  这可能需要几分钟..." -ForegroundColor Gray
Write-Host ""

docker exec $CONTAINER_NAME bash -c "cd /sgl-workspace/sglang && pip install -e 'python[all]' --quiet"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] 安装失败" -ForegroundColor Red
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    exit 1
}
Write-Host "[安装] ✅ 安装完成" -ForegroundColor Green
Write-Host ""

# 运行测试
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "运行测试（修复后）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "预期日志:" -ForegroundColor Yellow
Write-Host "  TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend..." -ForegroundColor Green
Write-Host ""
Write-Host "按 Ctrl+C 可以中断" -ForegroundColor Gray
Write-Host ""

# 运行测试命令
docker exec -it $CONTAINER_NAME python3 -m sglang.launch_server `
    --model-path $MODEL_PATH `
    --tp 2 `
    --ep 2 `
    --port $PORT `
    --trust-remote-code `
    --enable-dp-attention `
    --disable-custom-all-reduce `
    --enable-two-batch-overlap `
    --moe-a2a-backend deepep `
    --log-level info

Write-Host ""
Write-Host "测试完成！" -ForegroundColor Green
Write-Host ""
Write-Host "清理容器..." -ForegroundColor Yellow
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
Write-Host "清理完成" -ForegroundColor Green
