import os, time

# USER CHANGABLE ARGUMENTS

# Change to `True` if you wish to enable these common arguments
extra_models_cpu = False
inbrowser = False
optimized_turbo = False
optimized = False
share = False

# Enter other `--arguments` you wish to use
other_arguments = ""





# BEGIN RELAUNCHER PYTHON CODE

common_arguments = ""

if extra_models_cpu == True:
    common_arguments += "--extra-models-cpu "
if optimized_turbo == True:
    common_arguments += "--optimized-turbo "
if optimized == True:
    common_arguments += "--optimized "
if share == True:
    common_arguments += "--share "

if inbrowser == True:
    inbrowser_argument = "--inbrowser "
else:
    inbrowser_argument = ""

n = 0
while True:
    if n == 0:
        print('Relauncher: Launching...')
        os.system(f"python scripts/webui.py {common_arguments} {inbrowser_argument} {other_arguments}")
        
    else:
        print(f'\tRelaunch count: {n}')
        print('Relauncher: Launching...')
        os.system(f"python scripts/webui.py {common_arguments} {other_arguments}")
    
    print('Relauncher: Process is ending. Relaunching in 1s...')
    n += 1
    time.sleep(1)
