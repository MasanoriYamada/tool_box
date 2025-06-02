from __future__ import print_function
import threading
from joblib import Parallel, delayed
import queue
import os
import sys
import time
import subprocess
try:
    import torch
except ImportError:
    pass

# Common settings
TASK = 0
N_GPU = 3  # max gpu in machine

# Task queue
cmd_list = []

if TASK == 0:
    # 会話生成と評価のタスクを追加
    # base vs base
    cmd_list.append('uv run python example.py --seed 0')
else:
    print('no task')

print(f'task No: {TASK}')
print(f'Number of jobs: {len(cmd_list)}')

try:
    n_actual_gpu = torch.cuda.device_count()
except (ImportError, AttributeError):
    n_actual_gpu = 0

for cmd in cmd_list:
    print(cmd)
print(f'Total GPUS: {n_actual_gpu}')
print('run? [Y/N]')
y_n = str(input()).upper()
if y_n != 'Y':
    print('terminate')
    sys.exit()

_print = print
_rlock = threading.RLock()

def print(*args, **kwargs):
    with _rlock:
        _print(*args, **kwargs)

q = queue.Queue(maxsize=N_GPU)
for i in range(N_GPU):
    q.put(i)

def runner(i, cmd):
    gpu = q.get() % N_GPU
    global n_actual_gpu
    if n_actual_gpu == 0:
        n_actual_gpu = 1
    act_gpu = gpu % n_actual_gpu
    print(f'In gpu: {act_gpu}, cmd: {cmd}')
    time.sleep(gpu * 5)  # for avoiding conflict
    subprocess.run(f"CUDA_VISIBLE_DEVICES={act_gpu} {cmd}", shell=True, check=True)
    q.put(gpu)

# Job start
Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(i, cmd) for i, cmd in enumerate(cmd_list))
