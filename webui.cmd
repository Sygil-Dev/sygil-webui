@echo off
:: This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

:: Copyright 2022 sd-webui team.
:: This program is free software: you can redistribute it and/or modify
:: it under the terms of the GNU Affero General Public License as published by
:: the Free Software Foundation, either version 3 of the License, or
:: (at your option) any later version.

:: This program is distributed in the hope that it will be useful,
:: but WITHOUT ANY WARRANTY; without even the implied warranty of
:: MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
:: GNU Affero General Public License for more details.

:: You should have received a copy of the GNU Affero General Public License
:: along with this program.  If not, see <http://www.gnu.org/licenses/>. 
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
echo Stashing local changes and pulling latest update...
call git stash
call git pull
set /P restore="Do you want to restore changes you made before updating? (Y/N): "
IF /I "%restore%" == "N" (
  echo Removing changes please wait...
  call git stash drop
  echo Changes removed, press any key to continue...
  pause >nul
) ELSE IF /I "%restore%" == "Y" (
  echo Restoring changes, please wait...
  call git stash pop --quiet
  echo Changes restored, press any key to continue...
  pause >nul
)
call "%v_conda_path%\Scripts\activate.bat"

for /f "delims=" %%a in ('git log -1 --format^="%%H" -- environment.yaml')  DO set v_cur_hash=%%a
set /p "v_last_hash="<"z_version_env.tmp"
echo %v_cur_hash%>z_version_env.tmp

echo Current  environment.yaml hash: %v_cur_hash%
echo Previous environment.yaml hash: %v_last_hash%

if "%v_last_hash%" == "%v_cur_hash%" (
  echo environment.yaml unchanged. dependencies should be up to date.
  echo if you still have unresolved dependencies, delete "z_version_env.tmp"
) else (
  echo environment.yaml changed. updating dependencies
  call conda env create --name "%v_conda_env_name%" -f environment.yaml
  call conda env update --name "%v_conda_env_name%" -f environment.yaml
)


call "%v_conda_path%\Scripts\activate.bat" "%v_conda_env_name%"

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "models\ldm\stable-diffusion-v1\model.ckpt" (
  set "PYTHONPATH=%~dp0"
  python scripts\relauncher.py %*
) ELSE (
  echo Your model file does not exist! Place it in 'models\ldm\stable-diffusion-v1' with the name 'model.ckpt'.
  pause
)

::cmd /k
