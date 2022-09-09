@echo off

:: copy over the first line from environment.yaml, e.g. name: ldm, and take the second word after splitting by ":" delimiter 
set /p first_line=< environment.yaml  
for /f "tokens=2 delims=:" %%i in ("%first_line%") do set untrimmed_conda_env_name=%%i
:: remove whitespaces
for /f "tokens=* delims= " %%a in ("%untrimmed_conda_env_name%") do set conda_env_name=%%a
echo Environment name is set as %conda_env_name% as per environment.yaml

:: Put the path to conda directory after "=" sign if it's installed at non-standard path:
set custom_conda_path=

IF NOT "%custom_conda_path%"=="" (
  set paths=%custom_conda_path%;%paths%
)
:: Put the path to conda directory in a file called "custom-conda-path.txt" if it's installed at non-standard path:
FOR /F %%i IN (custom-conda-path.txt) DO set custom_conda_path=%%i

set paths=%ProgramData%\miniconda3
set paths=%paths%;%USERPROFILE%\miniconda3
set paths=%paths%;%ProgramData%\anaconda3
set paths=%paths%;%USERPROFILE%\anaconda3

for %%a in (%paths%) do (
 IF NOT "%custom_conda_path%"=="" (
   set paths=%custom_conda_path%;%paths%
 )
)

for %%a in (%paths%) do ( 
 if EXIST "%%a\Scripts\activate.bat" (
    SET CONDA_PATH=%%a
    echo anaconda3/miniconda3 detected in %%a
    goto :foundPath
 )
)

IF "%CONDA_PATH%"=="" (
  echo anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html
  exit /b 1
)

call git stash
call git pull

:foundPath
call "%CONDA_PATH%\Scripts\activate.bat"
call conda env create -n "%conda_env_name%" -f environment.yaml
call conda env update -n "%conda_env_name%" --file environment.yaml --prune
call "%CONDA_PATH%\Scripts\activate.bat" "%conda_env_name%"
python "%CD%"\scripts\relauncher.py

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "models\ldm\stable-diffusion-v1\model.ckpt" (
  python scripts/relauncher.py
) ELSE (
  ECHO Your model file does not exist! Place it in 'models\ldm\stable-diffusion-v1' with the name 'model.ckpt'.
)