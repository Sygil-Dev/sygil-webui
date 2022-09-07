@echo off
set conda_env_name=ldm

# Array of model files to pre-download
# local filename
# local path in container (no trailing slash)
# download URL
# sha256sum
MODEL_FILES=(
    'model.ckpt /sd/models/ldm/stable-diffusion-v1 https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
    'GFPGANv1.3.pth /sd/src/gfpgan/experiments/pretrained_models https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70'
    'RealESRGAN_x4plus.pth /sd/src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
    'RealESRGAN_x4plus_anime_6B.pth /sd/src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
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
call conda env create -n "%conda_env_name%" -f environment.yaml
call conda env update --name "%conda_env_name%" -f environment.yaml
call "%CONDA_PATH%\Scripts\activate.bat" "%conda_env_name%"
::python "%CD%"\scripts\relauncher.py

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "models\ldm\stable-diffusion-v1\model.ckpt" (
  python -m streamlit run scripts\webui_streamlit.py
) ELSE (
  ECHO Your model file does not exist! Place it in 'models\ldm\stable-diffusion-v1' with the name 'model.ckpt'.
)