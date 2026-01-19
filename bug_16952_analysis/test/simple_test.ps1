# 简单测试脚本 - 实时显示所有输出

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bug #16952 简单测试（实时输出）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查模型
$cachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
if (Test-Path $cachePath) {
    $files = Get-ChildItem $cachePath -Recurse -File -ErrorAction SilentlyContinue
    if ($files) {
        $sizeGB = [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
        Write-Host "✅ 模型已下载: $sizeGB GB" -ForegroundColor Green
    } else {
        Write-Host "⏳ 模型正在下载中..." -ForegroundColor Yellow
    }
} else {
    Write-Host "⏳ 模型未下载，将自动下载" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "运行测试命令..." -ForegroundColor Yellow
Write-Host "（首次运行会下载模型，可能需要 10-30 分钟）" -ForegroundColor Gray
Write-Host "（按 Ctrl+C 可以中断）" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 直接运行，实时显示输出
python3 -m sglang.launch_server `
    --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat `
    --tp 2 `
    --ep 2 `
    --port 32123 `
    --trust-remote-code `
    --enable-dp-attention `
    --disable-custom-all-reduce `
    --enable-two-batch-overlap `
    --moe-a2a-backend deepep `
    --log-level info

Write-Host ""
Write-Host "测试完成！" -ForegroundColor Green
