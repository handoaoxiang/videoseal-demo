param([switch]$WithVae, [switch]$WithCuda)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & powershell -ExecutionPolicy Bypass -File .\setup.ps1
    if ($LASTEXITCODE -ne 0) { throw '基础环境安装失败' }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
if ($LASTEXITCODE -ne 0) { throw '网页依赖安装失败' }
if ($WithCuda) {
    & .\.venv\Scripts\python.exe -m pip install 'torch==2.7.1+cu126' --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'CUDA 版 PyTorch 安装失败' }
    & .\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动'; print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA 运行检查失败；仍可在网页选择 CPU' }
}
if ($WithVae) {
    & .\.venv\Scripts\python.exe -m pip install -r requirements-vae.txt
    if ($LASTEXITCODE -ne 0) { throw 'VAE 依赖安装失败' }
    & .\.venv\Scripts\python.exe prepare_vae.py
    if ($LASTEXITCODE -ne 0) { throw 'VAE 模型准备失败' }
}
Write-Output '网页环境已就绪。双击 run_web.cmd 启动。'
