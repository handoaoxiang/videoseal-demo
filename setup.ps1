$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw '创建 Python 环境失败' }
}
& ./.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw '安装 PyTorch 失败' }
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '安装演示依赖失败' }
& ./.venv/Scripts/python.exe demo.py --download-model
if ($LASTEXITCODE -ne 0) { throw '下载官方模型失败' }
Write-Output '环境已就绪。运行 run_demo.cmd 打开演示。'
