@echo off

:: Run all commands using this script's directory as the working directory
cd %~dp0

:: copy over the first line from environment.yaml, e.g. name: ldm, and take the second word after splitting by ":" delimiter
for /F "tokens=2 delims=: " %%i in (environment.yaml) DO (
  set v_conda_env_name=%%i
  goto EOL
)
:EOL

echo Environment name is set as %v_conda_env_name% as per environment.yaml

:: Put the path to conda directory in a file called "custom-conda-path.txt" if it's installed at non-standard path
IF EXIST custom-conda-path.txt (
  FOR /F %%i IN (custom-conda-path.txt) DO set v_custom_path=%%i
)

set v_paths=%ProgramData%\miniconda3
set v_paths=%v_paths%;%USERPROFILE%\miniconda3
set v_paths=%v_paths%;%ProgramData%\anaconda3
set v_paths=%v_paths%;%USERPROFILE%\anaconda3

for %%a in (%v_paths%) do (
  IF NOT "%v_custom_path%"=="" (
    set v_paths=%v_custom_path%;%v_paths%
  )
)

for %%a in (%v_paths%) do (
  if EXIST "%%a\Scripts\activate.bat" (
    SET v_conda_path=%%a
    echo anaconda3/miniconda3 detected in %%a
    goto :CONDA_FOUND
  )
)

IF "%v_conda_path%"=="" (
  echo anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html
  pause
  exit /b 1
)

:CONDA_FOUND

if not exist "z_version_env.tmp" (
  :: first time running, we need to update
  set AUTO=1
  call "update_to_latest.cmd"
)

call "%v_conda_path%\Scripts\activate.bat" "%v_conda_env_name%"

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "models\ldm\stable-diffusion-v1\model.ckpt" (
  set PYTHONPATH=%~dp0
  python scripts\relauncher.py
) ELSE (
  echo Your model file does not exist! Place it in 'models\ldm\stable-diffusion-v1' with the name 'model.ckpt'.
  pause
)
