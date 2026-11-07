# 一键打包脚本
#   pwsh -File tools\build_exe.ps1              # 目录版（快，输出 dist\StockTrendAnalyzer\）
#   pwsh -File tools\build_exe.ps1 -Clean       # 清理后重新打包目录版
#   pwsh -File tools\build_exe.ps1 -OneFile     # 单文件版（发送给别人用，输出 dist-onefile\StockTrendAnalyzer.exe）
param([switch]$Clean, [switch]$OneFile)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$python = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    throw "未找到虚拟环境解释器：$python（请先在 F:\毕业设计 下创建 .venv 并安装 requirements.txt）"
}

$specName = if ($OneFile) { 'StockTrendAnalyzer-onefile.spec' } else { 'StockTrendAnalyzer.spec' }
$distName = if ($OneFile) { 'dist-onefile' } else { 'dist' }
$specPath = Join-Path $projectRoot $specName
if (-not (Test-Path $specPath)) { throw "未找到打包配置：$specPath" }

Push-Location $projectRoot
try {
    if ($Clean) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue 'build', 'dist', 'dist-onefile'
    }
    if (-not (Test-Path 'resources\app.ico')) {
        Write-Host '未找到 resources\app.ico，先自动生成图标…'
        & $python -X utf8 tools\make_icon.py (Join-Path $projectRoot 'resources\app.ico')
    }

    $mode = if ($OneFile) { '单文件版' } else { '目录版' }
    Write-Host "开始打包（$mode，首次约 1-3 分钟）…"
    & $python -m PyInstaller $specName --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败，退出码 $LASTEXITCODE" }

    if ($OneFile) {
        $target = Join-Path $projectRoot "$distName\StockTrendAnalyzer.exe"
        if (-not (Test-Path $target)) { throw "未生成可执行文件：$target" }
        $size = [math]::Round((Get-Item $target).Length / 1MB, 1)
        Write-Host ''
        Write-Host "打包完成：$target（$size MB，单个文件可直接发送）"
    } else {
        $target = Join-Path $projectRoot "$distName\StockTrendAnalyzer"
        $exe = Join-Path $target 'StockTrendAnalyzer.exe'
        if (-not (Test-Path $exe)) { throw "未生成可执行文件：$exe" }
        $size = [math]::Round((Get-ChildItem $target -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
        Write-Host ''
        Write-Host "打包完成：$exe"
        Write-Host "输出目录：$target（合计 $size MB，发送时请把整个文件夹压缩后发送）"
    }

    Write-Host "自检命令：& `"$target`" --self-test   # 生成 self_test_report.json"
} finally {
    Pop-Location
}
