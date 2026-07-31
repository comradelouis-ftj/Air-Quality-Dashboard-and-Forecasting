@echo off

:: Change directory to peoject directory
cd /d "%~dp0"

call ne_env\Scripts\activate

:: running python file
python extraction_uptodate_dataset.py

:: github update
git add datasets_aggr/weather_readings_mlready.csv
git commit -m "auto-update: updated dataset to current date"
git push -u origin master

call ne_env\Scripts\deactivate

powershell -Command "(New-Object -ComObject WScript.Shell).Popup('Data successfully extracted and pushed to GitHub!', 5, 'Streamlit Automation', 64)"