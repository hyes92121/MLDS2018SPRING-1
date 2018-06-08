#!/bin/bash 
wget -cO  https://www.dropbox.com/s/u7z71abtzfzexs0/model.tar?dl=0 > model.tar
python3 infer.py $1

