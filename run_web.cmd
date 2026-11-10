@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行 powershell -ExecutionPolicy Bypass -File .\setup_web.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, cv2, scipy, pywt"
if errorlevel 1 (
  echo 请先运行 powershell -ExecutionPolicy Bypass -File .\setup_web.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import json,urllib.request; data=json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8765/api/capabilities',timeout=1)); assert 'methods' in data" >nul 2>nul
if not errorlevel 1 (
  start "" "http://127.0.0.1:8765"
  exit /b 0
)
echo 频谱工坊：http://127.0.0.1:8765
echo 关闭此窗口会停止服务；正在执行的任务将在下次启动时标记为中断。
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m webapp.server
if errorlevel 1 pause
