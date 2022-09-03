#!/usr/bin/env python
# sync_local.py

import argparse, os, sys, glob, re, yaml, shutil

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dest",   type=str, help="dir to write files to. Required.", default=None)
parser.add_argument("-s", "--source", type=str, help="dir to read files from.", default=".")
parser.add_argument("-c", "--config", type=str, help="path to sync.yaml", default="./.github/sync.yml")
parser.add_argument("-k", "--key",   type=str, help="key to read from sync.yaml. (Default: hlky/stable-diffusion)", default="hlky/stable-diffusion")
parser.add_argument("-r", "--reverse", action='store_true', help="Swaps source and destination to do a reverse copy.")
opt = parser.parse_args()

#Validate required parameters
if opt.dest is None:
    raise SystemExit("Error: Missing --dest argument. Need to know where to copy files to.")

#Read config file
config_file = None
if opt.config is not None and os.path.isfile(opt.config):
    try:
        print(f"Reading copy list from {opt.config}")
        with open(opt.config, "r", encoding="utf8") as f:
            config_file = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error loading config file {opt.config}:", e, file=sys.stderr)
else:
    raise SystemExit(f"Error: config file not found. Config value: {opt.config}")

#Double check config just in case
if config_file is None:
    raise SystemExit("Unknown error why config_file not set")

#Pull key from config file
try:
    print(f"Using key: {opt.key}")
    copyList = config_file[opt.key]
except(KeyError) as e:
    raise SystemExit(f"KeyError: could not find repo in config file. Looking for: {opt.key}")


#Copy each item pair in config file.
for i, copyItem in enumerate(copyList):
    if(copyItem["dest"]) is None:
        print(f"skip copy item {i}, missing dest")
        continue
    if(copyItem["source"]) is None:
        print("skip copy item {i}, missing source")
        continue
    
    fullDest = os.path.join(opt.dest, copyItem["dest"])
    fullSource = os.path.join(opt.source, copyItem["source"])
    
    if opt.reverse:
        fullDest, fullSource = fullSource, fullDest
    
    try:
        print(f"Copying file from source: [{fullSource}] to dest: [{fullDest}]")
        os.makedirs(os.path.dirname(fullDest), exist_ok=True)
        shutil.copy2(fullSource, fullDest)
    except (OSError, IOError) as e:
        print(f"Error copying file.", e, file=sys.stderr)
        
print("Done copying files.")
