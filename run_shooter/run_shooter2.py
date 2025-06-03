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

# Task definitions
class Task:
    def __init__(self, cmd, dependencies=None):
        self.cmd = cmd
        self.dependencies = dependencies or []
        self.completed = False
        self.running = False
        self.failed = False
        self.ready = False  # タスクが実行可能かどうかのフラグ

# Task queue
tasks = []
task_queue = queue.Queue()  # 実行可能なタスクを管理するキュー
task_lock = threading.Lock()  # タスク状態の更新を保護するロック

if TASK == 0:
    # テスト用の簡単なタスク
    task_1 = Task('echo "Task 1"')
    task_2 = Task('echo "Task 2"')
    task_3 = Task('echo "Task 3"')
    
    # 依存関係のあるタスク
    task_4 = Task('echo "Task 4 (depends on 1,2,3)"', dependencies=[task_1, task_2, task_3])
    task_5 = Task('echo "Task 5 (depends on 1,2,3)"', dependencies=[task_1, task_2, task_3])
    task_6 = Task('echo "Task 6 (depends on 1,2,3)"', dependencies=[task_1, task_2, task_3])

    # タスクを実行順序に従って追加
    tasks.extend([task_1, task_2, task_3, task_4, task_5, task_6])

elif TASK == 1:
    # 基本的なタスク
    task_1 = Task('echo "Task 1"')
    task_2 = Task('echo "Task 2"')
    task_3 = Task('echo "Task 3"')
    
    # 中間レベルの依存タスク
    task_4 = Task('echo "Task 4 (depends on 1,2)"', dependencies=[task_1, task_2])
    task_5 = Task('echo "Task 5 (depends on 2,3)"', dependencies=[task_2, task_3])
    task_6 = Task('echo "Task 6 (depends on 1,3)"', dependencies=[task_1, task_3])
    
    # 複雑な依存関係を持つタスク
    task_7 = Task('echo "Task 7 (depends on 4,5)"', dependencies=[task_4, task_5])
    task_8 = Task('echo "Task 8 (depends on 5,6)"', dependencies=[task_5, task_6])
    task_9 = Task('echo "Task 9 (depends on 4,6)"', dependencies=[task_4, task_6])
    
    # 最終タスク（すべての依存タスクに依存）
    task_10 = Task('echo "Task 10 (depends on 7,8,9)"', dependencies=[task_7, task_8, task_9])

    # タスクを実行順序に従って追加
    tasks.extend([task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10])
else:
    print('no task')

print(f'task No: {TASK}')
print(f'Number of jobs: {len(tasks)}')

try:
    n_actual_gpu = torch.cuda.device_count()
except (ImportError, AttributeError):
    n_actual_gpu = 0

for task in tasks:
    print(f"Task: {task.cmd}")
    if task.dependencies:
        print(f"  Dependencies: {[dep.cmd for dep in task.dependencies]}")
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

def check_dependencies(task):
    """依存関係のチェック（ロックなし）"""
    if not task.dependencies:
        return True
    return all(dep.completed and not dep.failed for dep in task.dependencies)

def update_task_queue():
    # ロックを使わずに、putはロック外で行う
    to_queue = []
    with task_lock:
        for task in tasks:
            if (not task.completed and not task.failed and check_dependencies(task)):
                if not getattr(task, 'queued', False):
                    task.queued = True
                    to_queue.append(task)
    for task in to_queue:
        task_queue.put(task)
        print(f"Task queued: {task.cmd}")

def update_task_state(task, completed=False, failed=False):
    """タスクの状態を更新（改善版）"""
    with task_lock:
        task.completed = completed
        task.failed = failed
        task.running = False
        print(f"Task state updated: {task.cmd} (completed={completed}, failed={failed})")
        # 状態更新後に依存タスクをチェック
        update_task_queue()

def runner(i):
    while True:
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            # ロックはここで短く
            with task_lock:
                if all(task.completed or task.failed for task in tasks):
                    print(f"Worker {i} finished: all tasks completed or failed")
                    break
            update_task_queue()
            continue

        gpu = q.get() % N_GPU
        global n_actual_gpu
        if n_actual_gpu == 0:
            n_actual_gpu = 1
        act_gpu = gpu % n_actual_gpu
        print(f'Worker {i} starting task on GPU {act_gpu}: {task.cmd}')
        time.sleep(gpu * 5)
        try:
            result = subprocess.run(f"CUDA_VISIBLE_DEVICES={act_gpu} {task.cmd}",
                                    shell=True, check=True,
                                    capture_output=True, text=True)
            print(f"Task output: {result.stdout.strip()}")
            with task_lock:
                task.completed = True
                task.queued = False
                print(f"Task completed: {task.cmd}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing task: {task.cmd}")
            print(f"Error: {e}")
            with task_lock:
                task.failed = True
                task.queued = False
                print(f"Task failed: {task.cmd}")
        finally:
            update_task_queue()  # 完了後に新たなタスクをキューイング
            q.put(gpu)
            task_queue.task_done()

# 初期の実行可能なタスクをキューに追加
update_task_queue()

# Job start
Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(i) for i in range(N_GPU))

# 全タスクの完了を待機
while True:
    with task_lock:
        all_completed = all(task.completed or task.failed for task in tasks)
        if all_completed:
            print("All tasks completed or failed")
            break
    time.sleep(1)
