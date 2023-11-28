#!/bin/bash

PYTHONOPTIMIZE=1

pyinstaller \
    --noconfirm \
    --clean \
    --name="Hammerhead" \
    --add-data="./src/Hammerhead/assets:assets" \
    --add-data="./src/Hammerhead/resources:resources" \
    --collect-all="linear_operator" \
    --onedir \
    --windowed \
    ./src/Hammerhead/hammerhead.py


