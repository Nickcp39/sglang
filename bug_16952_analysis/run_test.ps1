# Bug #16952 快速测试脚本

param(
    [switch]$ApplyFix,
    [switch]$RunTest,
    [switch]$All
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bug #16952 修复测试脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$testDir = Join-Path $PSScriptRoot "test"

if ($All -or $ApplyFix) {
    Write-Host "1. 应用修复代码..." -ForegroundColor Yellow
    Set-Location $testDir
    python apply_fix.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "应用修复失败" -ForegroundColor Red
        Set-Location $PSScriptRoot
        exit 1
    }
    Set-Location $PSScriptRoot
    Write-Host ""
}

if ($All -or $RunTest) {
    Write-Host "2. 运行测试..." -ForegroundColor Yellow
    Set-Location $testDir
    python test_fix.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "测试失败" -ForegroundColor Red
        Set-Location $PSScriptRoot
        exit 1
    }
    Set-Location $PSScriptRoot
    Write-Host ""
}

if (-not $All -and -not $ApplyFix -and -not $RunTest) {
    Write-Host "用法:" -ForegroundColor Yellow
    Write-Host "  .\run_test.ps1 -ApplyFix    # 只应用修复"
    Write-Host "  .\run_test.ps1 -RunTest     # 只运行测试"
    Write-Host "  .\run_test.ps1 -All         # 应用修复并运行测试"
    Write-Host ""
    Write-Host "或者手动运行:" -ForegroundColor Yellow
    Write-Host "  cd test"
    Write-Host "  python apply_fix.py"
    Write-Host "  python test_fix.py"
}

Write-Host "完成！" -ForegroundColor Green
