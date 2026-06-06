@echo off
REM ============================================================================
REM  TSMAE Experiment Dashboard - Windows 더블클릭 바로가기 (WSL2)
REM  Windows 탐색기에서 이 파일을 더블클릭하면 WSL 안에서 대시보드를 띄우고
REM  기본 브라우저로 http://127.0.0.1:8000/ 를 엽니다.  콘솔 창에서 Ctrl-C 로 종료.
REM  포트 변경:  start_dashboard.bat 8123
REM ============================================================================
setlocal
set PORT=%1
if "%PORT%"=="" set PORT=8000
echo TSMAE Dashboard 시작 중... (포트 %PORT%)  -  종료하려면 이 창에서 Ctrl-C
wsl.exe bash -lc "cd /home/ykio/notebooks/TSMAE && ./run_dashboard.sh %PORT%"
pause
