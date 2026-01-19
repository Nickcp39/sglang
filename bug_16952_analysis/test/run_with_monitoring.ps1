# 带监控的测试运行脚本
# 实时显示模型下载进度和测试状态

$ErrorActionPreference = "Continue"

# 配置
$MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
$CACHE_PATH = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
$PORT = 32123

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bug #16952 测试（带实时监控）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查模型下载状态
Write-Host "[初始化] 检查模型下载状态..." -ForegroundColor Yellow
$modelDownloaded = $false
if (Test-Path $CACHE_PATH) {
    $files = Get-ChildItem $CACHE_PATH -Recurse -File -ErrorAction SilentlyContinue
    if ($files) {
        $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
        $sizeGB = [math]::Round($totalSize / 1GB, 2)
        Write-Host "[初始化] ✅ 模型已下载: $sizeGB GB ($($files.Count) 个文件)" -ForegroundColor Green
        $modelDownloaded = $true
    } else {
        Write-Host "[初始化] ⏳ 模型目录存在但文件为空，等待下载..." -ForegroundColor Yellow
    }
} else {
    Write-Host "[初始化] ⏳ 模型未下载，首次运行将自动下载" -ForegroundColor Yellow
}
Write-Host ""

# 启动监控任务（后台）
Write-Host "[监控] 启动下载进度监控..." -ForegroundColor Yellow
$monitorJob = Start-Job -ScriptBlock {
    param($cachePath)
    $lastSize = 0
    $lastFileCount = 0
    $noChangeCount = 0
    
    while ($true) {
        $timestamp = Get-Date -Format "HH:mm:ss"
        
        if (Test-Path $cachePath) {
            $files = Get-ChildItem $cachePath -Recurse -File -ErrorAction SilentlyContinue
            if ($files) {
                $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
                $sizeGB = [math]::Round($totalSize / 1GB, 2)
                $fileCount = $files.Count
                
                if ($totalSize -eq $lastSize -and $fileCount -eq $lastFileCount) {
                    $noChangeCount++
                } else {
                    $noChangeCount = 0
                    $speed = ""
                    if ($lastSize -gt 0) {
                        $sizeDiff = $totalSize - $lastSize
                        $speedMB = [math]::Round($sizeDiff / 1MB, 2)
                        $speed = " (+$speedMB MB/10s)"
                    }
                    Write-Output "[$timestamp] 📥 下载进度: $sizeGB GB ($fileCount 文件)$speed"
                }
                
                $lastSize = $totalSize
                $lastFileCount = $fileCount
            } else {
                Write-Output "[$timestamp] ⏳ 等待文件开始下载..."
            }
        } else {
            Write-Output "[$timestamp] ⏳ 等待下载开始..."
        }
        
        Start-Sleep -Seconds 10
    }
} -ArgumentList $CACHE_PATH

# 启动进程监控任务
Write-Host "[监控] 启动进程监控..." -ForegroundColor Yellow
$processJob = Start-Job -ScriptBlock {
    param($port)
    while ($true) {
        $timestamp = Get-Date -Format "HH:mm:ss"
        
        # 检查 Python 进程
        $pythonProcs = Get-Process python* -ErrorAction SilentlyContinue
        if ($pythonProcs) {
            $sglangProcs = $pythonProcs | Where-Object {
                try {
                    $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
                    $cmdLine -like "*sglang*"
                } catch { $false }
            }
            
            if ($sglangProcs) {
                foreach ($proc in $sglangProcs) {
                    $memMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
                    $cpu = [math]::Round($proc.CPU, 2)
                    Write-Output "[$timestamp] 🔄 进程运行中: PID $($proc.Id) | 内存: $memMB MB | CPU: $cpu"
                }
            }
        }
        
        # 检查端口
        $portConn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($portConn) {
            Write-Output "[$timestamp] ✅ 端口 $port 已监听（服务器可能已启动）"
        }
        
        Start-Sleep -Seconds 15
    }
} -ArgumentList $PORT

Write-Host "[监控] ✅ 监控已启动" -ForegroundColor Green
Write-Host ""

# 显示监控输出的函数
function Show-MonitorOutput {
    $monitorOutput = Receive-Job $monitorJob -ErrorAction SilentlyContinue
    $processOutput = Receive-Job $processJob -ErrorAction SilentlyContinue
    
    if ($monitorOutput) {
        $monitorOutput | ForEach-Object { Write-Host $_ -ForegroundColor Cyan }
    }
    
    if ($processOutput) {
        $processOutput | ForEach-Object { Write-Host $_ -ForegroundColor Green }
    }
}

# 运行测试命令
Write-Host "[测试] 启动测试命令..." -ForegroundColor Yellow
Write-Host "[测试] 命令: python3 -m sglang.launch_server --model-path $MODEL_PATH --tp 2 --ep 2 --port $PORT --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --log-level info" -ForegroundColor Gray
Write-Host ""
Write-Host "提示:" -ForegroundColor Yellow
Write-Host "  - 监控会在后台运行，显示下载和进程状态" -ForegroundColor Gray
Write-Host "  - 按 Ctrl+C 可以中断" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始运行..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 启动测试命令
$testProcess = Start-Process -FilePath "python3" -ArgumentList @(
    "-m", "sglang.launch_server",
    "--model-path", $MODEL_PATH,
    "--tp", "2",
    "--ep", "2",
    "--port", $PORT.ToString(),
    "--trust-remote-code",
    "--enable-dp-attention",
    "--disable-custom-all-reduce",
    "--enable-two-batch-overlap",
    "--moe-a2a-backend", "deepep",
    "--log-level", "info"
) -NoNewWindow -PassThru -RedirectStandardOutput "test_output.log" -RedirectStandardError "test_error.log"

Write-Host "[测试] ✅ 测试进程已启动 (PID: $($testProcess.Id))" -ForegroundColor Green
Write-Host "[测试] 输出日志: test_output.log" -ForegroundColor Gray
Write-Host "[测试] 错误日志: test_error.log" -ForegroundColor Gray
Write-Host ""

# 监控循环
try {
    $startTime = Get-Date
    while (-not $testProcess.HasExited) {
        # 显示监控输出
        Show-MonitorOutput
        
        # 显示最新的日志（最后5行）
        if (Test-Path "test_output.log") {
            $logLines = Get-Content "test_output.log" -Tail 3 -ErrorAction SilentlyContinue
            if ($logLines) {
                $logLines | ForEach-Object {
                    if ($_ -match "TBO/SBO|flashinfer_cutedsl|Automatically|Error|Exception") {
                        Write-Host "[日志] $_" -ForegroundColor Yellow
                    }
                }
            }
        }
        
        if (Test-Path "test_error.log") {
            $errorLines = Get-Content "test_error.log" -Tail 3 -ErrorAction SilentlyContinue
            if ($errorLines) {
                $errorLines | ForEach-Object {
                    Write-Host "[错误] $_" -ForegroundColor Red
                }
            }
        }
        
        $elapsed = (Get-Date) - $startTime
        Write-Host "[状态] 运行时间: $($elapsed.ToString('mm\:ss'))" -ForegroundColor Gray
        Write-Host ""
        
        Start-Sleep -Seconds 5
    }
    
    # 进程已退出
    Write-Host "[测试] 进程已退出，退出代码: $($testProcess.ExitCode)" -ForegroundColor Yellow
    
} catch {
    Write-Host "[错误] 监控出错: $_" -ForegroundColor Red
} finally {
    # 清理
    Write-Host ""
    Write-Host "[清理] 停止监控任务..." -ForegroundColor Yellow
    Stop-Job $monitorJob, $processJob -ErrorAction SilentlyContinue
    Remove-Job $monitorJob, $processJob -ErrorAction SilentlyContinue
    
    # 显示最终日志
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "最终日志（最后20行）" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (Test-Path "test_output.log") {
        Write-Host "标准输出:" -ForegroundColor Yellow
        Get-Content "test_output.log" -Tail 20 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    }
    
    if (Test-Path "test_error.log") {
        Write-Host "错误输出:" -ForegroundColor Yellow
        Get-Content "test_error.log" -Tail 20 | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }
}

Write-Host ""
Write-Host "完成！" -ForegroundColor Green
