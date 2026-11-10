@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行 setup.ps1 安装演示环境。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "demo.py"
if errorlevel 1 pause
