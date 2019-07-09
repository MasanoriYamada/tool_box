from __future__ import print_function
import threading
from joblib import Parallel, delayed
import queue
import os
import time
import subprocess

### common setting
task = 0
N_GPU = 2  # max gpu in machine
repeats = 5
epoch = 5000
test = 10
dataset = 'mnist'

##################### task queue
cmd_list = []


if task == 0:
    description = 'search learning rae'

    lrs = [0.001, 0.001, 0.01, 0.1]
    save_dir = 'log/' + dataset + '/vae'
    for repeat in range(repeats):
        for lr in lrs:
            tmp = 'python main.py --data {} --epochs {} --dir {} --seed {} --test {} --lr {} --description \'{}\''.format(
                dataset, epoch, save_dir, repeat, test, lr, description)
            cmd_list.append(tmp)

else:
    print('no task')

#####################
os.makedirs(save_dir, exist_ok=True)
_print = print
_rlock = threading.RLock()

def print(*args, **kwargs):
    with _rlock:
        _print(*args, **kwargs)
        
q = queue.Queue(maxsize=N_GPU)
for i in range(N_GPU):
    q.put(i)

def runner(i, cmd):
    gpu = q.get()
    print('In gpu: {}, cmd: {}'.format(gpu, cmd))
    time.sleep(i*5)  # for avoinding conflict
    subprocess.run("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd), shell=True, check=True)
    q.put(gpu)

# job start
Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(i, cmd) for i, cmd in enumerate(cmd_list))
