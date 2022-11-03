@echo off
:: This file is part of sygil-webui (https://github.com/Sygil-Dev/sygil-webui/).
:: 
:: Copyright 2022 Sygil-Dev team.
:: This program is free software: you can redistribute it and/or modify
:: it under the terms of the GNU Affero General Public License as published by
:: the Free Software Foundation, either version 3 of the License, or
:: (at your option) any later version.
:: 
:: This program is distributed in the hope that it will be useful,
:: but WITHOUT ANY WARRANTY; without even the implied warranty of
:: MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
:: GNU Affero General Public License for more details.
:: 
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

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%INSTALL_ENV_DIR%\Library\usr\bin;%PATH%

set v_paths=%INSTALL_ENV_DIR%
set v_paths=%v_paths%;%ProgramData%\miniconda3
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
git status --porcelain=1 -uno | findstr . && set "HasChanges=1" || set "HasChanges=0"
call git stash
call git pull
IF "%HasChanges%" == "0" GOTO SKIP_RESTORE

set /P restore="Do you want to restore changes you made before updating? (Y/N): "
IF /I "%restore%" == "N" (
  echo Removing changes...
  call git stash drop
  echo Changes removed
) ELSE IF /I "%restore%" == "Y" (
  echo Restoring changes...
  call git stash pop --quiet
  echo Changes restored
)

:SKIP_RESTORE
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
IF EXIST "models\ldm\stable-diffusion-v1\Stable Diffusion v1.5.ckpt" (
  python -m streamlit run scripts\webui_streamlit.py --theme.base dark --server.address localhost
) ELSE (
  echo Your model file does not exist! Once the WebUI launches please visit the Model Manager page and download the models by using the Download button for each model.
  python -m streamlit run scripts\webui_streamlit.py --theme.base dark --server.address localhost
)

::cmd /k
