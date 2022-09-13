%ProgramData%\miniconda3\scripts\activate.bat
set conda_env_name=ldm
conda env create -n "%conda_env_name%" -f environment.yaml
%ProgramData%\miniconda3\scripts\activate.bat "%conda_env_name%"