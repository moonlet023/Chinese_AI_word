@echo off
cd /d "%~dp0"

python main.py %*
python train_five_words.py %*