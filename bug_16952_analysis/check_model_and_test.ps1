# Bug #16952 模型检查和测试脚本
# 用于检查模型下载状态并运行测试

param(
    [string]$ModelPath = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    [int]$TpSize = 2,
    [int]$EpSize = 2,
    [int]$Port = 32123,
    [switch]$UseDocker = $false,
    [switch]$WithBackend = $false,
    [switch]$CheckOnly = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bug #16952 模型检查和测试脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查模型下载状态
Write-Host "1. 检查模型下载状态..." -ForegroundColor Yellow
$modelName = $ModelPath -replace "/", "--"
$cachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--$modelName"

if (Test-Path $cachePath) {
    Write-Host "   ✓ 找到模型缓存目录: $cachePath" -ForegroundColor Green
    
    # 检查是否有 snapshots 目录
    $snapshotsPath = Join-Path $cachePath "snapshots"
    if (Test-Path $snapshotsPath) {
        $snapshots = Get-ChildItem $snapshotsPath -Directory
        if ($snapshots.Count -gt 0) {
            $latestSnapshot = $snapshots | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            Write-Host "   ✓ 找到模型快照: $($latestSnapshot.Name)" -ForegroundColor Green
            
            # 检查模型文件
            $modelFiles = Get-ChildItem $latestSnapshot.FullName -Recurse -File | Measure-Object -Property Length -Sum
            $totalSizeGB = [math]::Round($modelFiles.Sum / 1GB, 2)
            Write-Host "   ✓ 模型文件总大小: $totalSizeGB GB" -ForegroundColor Green
            Write-Host "   ✓ 文件数量: $($modelFiles.Count)" -ForegroundColor Green
            
            # 检查关键文件
            $keyFiles = @("config.json", "tokenizer.json", "model.safetensors.index.json")
            $missingFiles = @()
            foreach ($file in $keyFiles) {
                $found = Get-ChildItem $latestSnapshot.FullName -Recurse -Filter $file -ErrorAction SilentlyContinue
                if (-not $found) {
                    $missingFiles += $file
                }
            }
            
            if ($missingFiles.Count -eq 0) {
                Write-Host "   ✓ 关键文件完整" -ForegroundColor Green
            } else {
                Write-Host "   ⚠ 缺少文件: $($missingFiles -join ', ')" -ForegroundColor Yellow
            }
        } else {
            Write-Host "   ⚠ 快照目录为空，模型可能正在下载..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ⚠ 未找到快照目录，模型可能正在下载..." -ForegroundColor Yellow
    }
} else {
    Write-Host "   ✗ 未找到模型缓存目录" -ForegroundColor Red
    Write-Host "   模型路径: $cachePath" -ForegroundColor Gray
    Write-Host "   首次运行将自动下载模型" -ForegroundColor Yellow
}

Write-Host ""

# 2. 检查 GPU
Write-Host "2. 检查 GPU 状态..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>$null
    if ($gpuInfo) {
        Write-Host "   ✓ GPU 信息:" -ForegroundColor Green
        $gpuInfo | ForEach-Object {
            $parts = $_ -split ","
            Write-Host "     名称: $($parts[0].Trim())" -ForegroundColor Gray
            Write-Host "     总显存: $($parts[1].Trim())" -ForegroundColor Gray
            Write-Host "     可用显存: $($parts[2].Trim())" -ForegroundColor Gray
        }
    } else {
        Write-Host "   ⚠ 无法获取 GPU 信息（可能没有 NVIDIA GPU 或驱动未安装）" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠ 无法检查 GPU: $_" -ForegroundColor Yellow
}

Write-Host ""

# 3. 检查端口
Write-Host "3. 检查端口 $Port 是否被占用..." -ForegroundColor Yellow
$portInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "   ⚠ 端口 $Port 已被占用" -ForegroundColor Yellow
    Write-Host "   进程 ID: $($portInUse.OwningProcess)" -ForegroundColor Gray
} else {
    Write-Host "   ✓ 端口 $Port 可用" -ForegroundColor Green
}

Write-Host ""

# 如果只是检查，就退出
if ($CheckOnly) {
    Write-Host "检查完成！" -ForegroundColor Cyan
    exit 0
}

# 4. 构建命令
Write-Host "4. 构建启动命令..." -ForegroundColor Yellow

$baseArgs = @(
    "--model-path", $ModelPath,
    "--tp", $TpSize.ToString(),
    "--ep", $EpSize.ToString(),
    "--port", $Port.ToString(),
    "--trust-remote-code",
    "--enable-dp-attention",
    "--disable-custom-all-reduce",
    "--enable-two-batch-overlap",
    "--moe-a2a-backend", "deepep"
)

if ($WithBackend) {
    $baseArgs += "--moe-runner-backend", "flashinfer_cutedsl"
    Write-Host "   ✓ 测试 2: 添加了 --moe-runner-backend flashinfer_cutedsl" -ForegroundColor Green
    Write-Host "   预期: 应该能正常运行" -ForegroundColor Green
} else {
    Write-Host "   ✓ 测试 1: 缺少 --moe-runner-backend flashinfer_cutedsl" -ForegroundColor Yellow
    Write-Host "   预期: 应该报错 AssertionError" -ForegroundColor Yellow
}

Write-Host ""

# 5. 选择运行方式
if ($UseDocker) {
    Write-Host "5. 使用 Docker 运行..." -ForegroundColor Yellow
    Write-Host "   注意: Docker 命令需要根据你的 Docker 配置调整" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Docker 命令（示例）:" -ForegroundColor Cyan
    Write-Host "docker run --gpus all -it --rm \" -ForegroundColor Gray
    Write-Host "  -v `"$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface`" \" -ForegroundColor Gray
    Write-Host "  -p $Port`:$Port \" -ForegroundColor Gray
    Write-Host "  lmsysorg/sglang:latest \" -ForegroundColor Gray
    Write-Host "  python3 -m sglang.launch_server $($baseArgs -join ' ')" -ForegroundColor Gray
    Write-Host ""
    
    $runDocker = Read-Host "是否运行 Docker 命令? (y/n)"
    if ($runDocker -eq "y") {
        Write-Host "运行 Docker 命令..." -ForegroundColor Yellow
        # 这里可以添加实际的 Docker 运行命令
        Write-Host "（需要根据你的 Docker 配置调整）" -ForegroundColor Gray
    }
} else {
    Write-Host "5. 运行 Python 命令..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "执行命令:" -ForegroundColor Cyan
    $command = "python3 -m sglang.launch_server $($baseArgs -join ' ')"
    Write-Host $command -ForegroundColor Gray
    Write-Host ""
    
    $runCommand = Read-Host "是否运行命令? (y/n)"
    if ($runCommand -eq "y") {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "开始运行..." -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "提示:" -ForegroundColor Yellow
        Write-Host "  - 首次运行会下载模型，可能需要几分钟到几十分钟" -ForegroundColor Gray
        Write-Host "  - 下载时可能没有输出，这是正常的" -ForegroundColor Gray
        Write-Host "  - 按 Ctrl+C 可以中断" -ForegroundColor Gray
        Write-Host ""
        
        # 运行命令
        & python3 -m sglang.launch_server $baseArgs
    }
}

Write-Host ""
Write-Host "完成！" -ForegroundColor Cyan
