#!/bin/bash

sudo apt install python3.8
pip install --upgrade pip
pip install -r requirements.txt
pip install pyvirtualdisplay
pip install tqdm

des=`python ../utils/print_game_lib_folder.py`
cp -f ../data/* $des/.
