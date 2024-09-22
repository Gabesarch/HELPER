#!/usr/bin/env pythons
import subprocess
import shlex
import re
import platform
import tempfile
import os
import sys

def get_current_busid():
    command = shlex.split('nvidia-smi -a')
    output = subprocess.check_output(command).decode()
    bus_id = output.split('Bus Id                            : ')[-1].split('\n')[0]
    return bus_id

def get_display_number():
    command = shlex.split('nvidia-smi -a')
    output = subprocess.check_output(command).decode()
    gpu_idx = int(output.split('Minor Number                          : ')[-1].split('\n')[0])
    return gpu_idx

def _startx(gpu_idx):

    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    print("GPU IDX is ", gpu_idx)

    try:
        command = shlex.split(f'X :{gpu_idx} -layout "X.org Configured" -screen Screen{gpu_idx} -sharevts')
        proc = subprocess.Popen(command)
    except Exception as e:
        print(e)

    return gpu_idx

def startx(display=None):
    gpu_idx = get_display_number() 

    _startx(gpu_idx)

    return str(gpu_idx)

if __name__ == "__main__":
    gpu_idx = None
    if len(sys.argv) > 1:
        gpu_idx = int(sys.argv[1])
    startx(gpu_idx)