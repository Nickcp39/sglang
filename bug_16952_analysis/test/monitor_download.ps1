# 监控模型下载进度

Write-Host "监控模型下载进度..." -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止监控" -ForegroundColor Yellow
Write-Host ""

$cachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat"

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
            
            # 检查是否有变化
            if ($totalSize -eq $lastSize -and $fileCount -eq $lastFileCount) {
                $noChangeCount++
                if ($noChangeCount -gt 6) {
                    Write-Host "[$timestamp] ⚠️  下载似乎已停止（30秒无变化）" -ForegroundColor Yellow
                    $noChangeCount = 0
                }
            } else {
                $noChangeCount = 0
                $speed = ""
                if ($lastSize -gt 0) {
                    $sizeDiff = $totalSize - $lastSize
                    $speedMB = [math]::Round($sizeDiff / 1MB, 2)
                    $speed = " (+$speedMB MB/10s)"
                }
                Write-Host "[$timestamp] ✅ 已下载: $sizeGB GB ($fileCount 个文件)$speed" -ForegroundColor Green
            }
            
            $lastSize = $totalSize
            $lastFileCount = $fileCount
        } else {
            Write-Host "[$timestamp] ⏳ 等待文件开始下载..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[$timestamp] ⏳ 等待下载开始..." -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 10
}
