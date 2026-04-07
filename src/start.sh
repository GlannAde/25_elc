#!/bin/bash
echo ";'" | sudo -S chmod 777 /home/elc/src/start.sh
echo ";'" | sudo -S chmod 777 /home/elc/src/main.py
v4l2-ctl --list-devices  # 列出所有视频设备
# sudo rdk-miniboot-update
/usr/bin/python3 /home/opi5max/elc/src/main.py