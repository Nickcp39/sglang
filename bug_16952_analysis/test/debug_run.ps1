# 调试运行脚本 - 显示详细输出

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bug #16952 调试运行脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 进程
Write-Host "1. 检查现有 Python 进程..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "   找到 Python 进程:" -ForegroundColor Yellow
    $pythonProcesses | ForEach-Object {
        Write-Host "   - PID: $($_.Id), 内存: $([math]::Round($_.WorkingSet64 / 1MB, 2)) MB" -ForegroundColor Gray
    }
} else {
    Write-Host "   没有运行中的 Python 进程" -ForegroundColor Green
}
Write-Host ""

# 检查 GPU
Write-Host "2. 检查 GPU 状态..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>$null
    if ($gpuInfo) {
        Write-Host "   GPU 状态:" -ForegroundColor Green
        $gpuInfo | ForEach-Object {
            $parts = $_ -split ","
            Write-Host "   - $($parts[0].Trim()): 显存 $($parts[1].Trim())/$($parts[2].Trim()), 利用率 $($parts[3].Trim())" -ForegroundColor Gray
        }
    } else {
        Write-Host "   无法获取 GPU 信息" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   无法检查 GPU: $_" -ForegroundColor Yellow
}
Write-Host ""

# 检查模型下载状态
Write-Host "3. 检查模型下载状态..." -ForegroundColor Yellow
$cachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
if (Test-Path $cachePath) {
    $files = Get-ChildItem $cachePath -Recurse -File -ErrorAction SilentlyContinue
    if ($files) {
        $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
        $sizeGB = [math]::Round($totalSize / 1GB, 2)
        Write-Host "   模型已下载: $sizeGB GB ($($files.Count) 个文件)" -ForegroundColor Green
    } else {
        Write-Host "   模型目录存在但文件为空，可能正在下载..." -ForegroundColor Yellow
    }
} else {
    Write-Host "   模型未下载，首次运行将自动下载" -ForegroundColor Yellow
}
Write-Host ""

# 运行命令（带详细输出）
Write-Host "4. 运行测试命令（带详细日志）..." -ForegroundColor Yellow
Write-Host "   命令: python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info" -ForegroundColor Gray
Write-Host ""
Write-Host "提示:" -ForegroundColor Yellow
Write-Host "  - 首次运行会下载模型（可能需要 10-30 分钟）" -ForegroundColor Gray
Write-Host "  - 下载时可能没有输出，这是正常的" -ForegroundColor Gray
Write-Host "  - 按 Ctrl+C 可以中断" -ForegroundColor Gray
Write-Host "  - 在另一个终端运行监控脚本查看下载进度" -ForegroundColor Gray
Write-Host ""
Write-Host "开始运行..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$cmd = @(
    "python3", "-m", "sglang.launch_server",
    "--model-path", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "--tp", "2",
    "--ep", "2",
    "--port", "32123",
    "--trust-remote-code",
    "--enable-dp-attention",
    "--disable-custom-all-reduce",
    "--enable-two-batch-overlap",
    "--moe-a2a-backend", "deepep",
    "--log-level", "info"
)

# 运行命令，实时显示输出
& $cmd[0] $cmd[1..($cmd.Length-1)]
