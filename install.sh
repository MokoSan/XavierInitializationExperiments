#!/bin/bash

echo "~ C1 Secret ~"

echo "Creating Virtual Env"
virtualenv -p python2 env

echo "Activating Virtual Env"
source ./env/bin/activate

echo "Installing Requirements"
pip install -r requirements.txt

echo "Adding repos to lib"
mkdir lib
cd lib
git clone https://github.com/anmolsjoshi/KerasHelpers.git
git clone https://github.com/glorotxa/Shapeset.git
cp KerasHelpers/__init__.py ..
cp KerasHelpers/__init__.py Shapeset/