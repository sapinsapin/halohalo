#!/bin/bash
sudo apt install -y python3.12-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
