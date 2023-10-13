import subprocess
import time
import shlex
import re
import atexit
import platform
import tempfile
import threading
import os
import sys
import random

import ipdb
st = ipdb.set_trace

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
    # NOTE: DISPLAY IS NOT USED HERE

    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    print("GPU IDX is ", gpu_idx)

    try:
        command = shlex.split(f'X :{gpu_idx} -layout "X.org Configured" -screen Screen{gpu_idx} -sharevts')
        proc = subprocess.Popen(command)
        atexit.register(lambda: proc.poll() is None and proc.kill())
        proc.wait()
    except Exception as e:
        print(e)

    return gpu_idx

def startx(display=None):
    gpu_idx = get_display_number() # this is used as display number

    xthread = threading.Thread(target=_startx, args=(gpu_idx,))
    xthread.daemon = True
    xthread.start()
    # wait for server to start
    time.sleep(4)

    return str(gpu_idx)


if __name__ == "__main__":
    startx()



    

