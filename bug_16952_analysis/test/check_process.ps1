# 检查进程状态

Write-Host "检查 sglang 进程状态..." -ForegroundColor Cyan
Write-Host ""

# 检查 Python 进程
$pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "找到 Python 进程:" -ForegroundColor Green
    foreach ($proc in $pythonProcesses) {
        $memMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
        $cpu = $proc.CPU
        Write-Host "  PID: $($proc.Id) | 内存: $memMB MB | CPU: $cpu" -ForegroundColor Gray
        
        # 检查命令行参数
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            if ($cmdLine -like "*sglang*") {
                Write-Host "   命令: $cmdLine" -ForegroundColor Cyan
            }
        } catch {
            # 忽略错误
        }
    }
} else {
    Write-Host "没有运行中的 Python 进程" -ForegroundColor Yellow
}

Write-Host ""

# 检查端口
Write-Host "检查端口 32123..." -ForegroundColor Cyan
$portConn = Get-NetTCPConnection -LocalPort 32123 -ErrorAction SilentlyContinue
if ($portConn) {
    Write-Host "  端口 32123 已被占用" -ForegroundColor Yellow
    Write-Host "  进程 ID: $($portConn.OwningProcess)" -ForegroundColor Gray
} else {
    Write-Host "  端口 32123 可用" -ForegroundColor Green
}

Write-Host ""

# 检查 GPU
Write-Host "检查 GPU 使用情况..." -ForegroundColor Cyan
try {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
} catch {
    Write-Host "  无法获取 GPU 信息" -ForegroundColor Yellow
}
