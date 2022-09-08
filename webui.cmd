@echo off

set conda_env_name=ldm
set conda_command=conda

set use_mamba_instead_of_conda=1
if "%use_mamba_instead_of_conda%"=="1" (
  where /q mamba
  if errorlevel 0 echo mamba installed
  if errorlevel 1 conda install mamba -n base -c conda-forge
  set conda_command=mamba
)

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

:foundPath
call "%CONDA_PATH%\Scripts\activate.bat"
%conda_command% env list | findstr /r /b /c:"%conda_env_name%"
if errorlevel 0 goto :conda_env_update
if errorlevel 1 goto :conda_env_create
goto :eof

:conda_env_create
call %conda_command% env create --name "%conda_env_name%" --file environment.yaml
goto :conda_activate

:conda_env_update
call %conda_command% env update --name "%conda_env_name%" --file environment.yaml --prune
goto :conda_activate

:conda_activate
call "%CONDA_PATH%\Scripts\activate.bat" "%conda_env_name%"
python "%CD%"\scripts\relauncher.py
goto :eof