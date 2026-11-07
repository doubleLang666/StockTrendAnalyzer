# 一键打包脚本：在项目根目录执行
#   pwsh -File tools\build_exe.ps1            # 增量打包
#   pwsh -File tools\build_exe.ps1 -Clean     # 先清理再打包
param([switch]$Clean)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$python = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    throw "未找到虚拟环境解释器：$python（请先在 F:\毕业设计 下创建 .venv 并安装 requirements.txt）"
}
if (-not (Test-Path (Join-Path $projectRoot 'StockTrendAnalyzer.spec'))) {
    throw "未找到打包配置：$projectRoot\StockTrendAnalyzer.spec"
}

Push-Location $projectRoot
try {
    if ($Clean) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue 'build', 'dist'
    }
    if (-not (Test-Path 'resources\app.ico')) {
        Write-Host '未找到 resources\app.ico，先自动生成图标…'
        & $python -X utf8 tools\make_icon.py (Join-Path $projectRoot 'resources\app.ico')
    }

    Write-Host '开始打包（首次约 1-3 分钟）…'
    & $python -m PyInstaller StockTrendAnalyzer.spec --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败，退出码 $LASTEXITCODE" }

    $dist = Join-Path $projectRoot 'dist\StockTrendAnalyzer'
    $exe = Join-Path $dist 'StockTrendAnalyzer.exe'
    if (-not (Test-Path $exe)) { throw "未生成可执行文件：$exe" }

    $size = [math]::Round((Get-ChildItem $dist -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
    Write-Host ''
    Write-Host "打包完成：$exe"
    Write-Host "输出目录：$dist（合计 $size MB）"
    Write-Host "自检命令：& `"$exe`" --self-test   # 生成 self_test_report.json"
} finally {
    Pop-Location
}
