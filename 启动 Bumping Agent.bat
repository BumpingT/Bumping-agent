@echo off
cd /d "C:\Users\Administrator\Documents\ai agent"
echo Starting Bumping Agent...
echo Open http://localhost:5003 in your browser
echo.
start "" "http://localhost:5003"
"C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" _run_5003.py
echo.
pause
