#!/bin/bash

# This creates the installer zip files that will be distributed to users
# It packs install.{sh,bat} along with a readme, and ensures that the user
# has the install script inside a new empty folder (after unzipping),
# otherwise the git repo will extract into whatever folder the script is in.

cd "$(dirname "${BASH_SOURCE[0]}")"

# make the installer zip for linux and mac
rm -rf sygil
mkdir -p sygil
cp install.sh sygil
cp readme.txt sygil

zip -r sygil-linux.zip sygil
zip -r sygil-mac.zip sygil

# make the installer zip for windows
rm -rf sygil
mkdir -p sygil
cp install.bat sygil
cp readme.txt sygil

zip -r sygil-windows.zip sygil

echo "The installer zips are ready to be distributed.."
