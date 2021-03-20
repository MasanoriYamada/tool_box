import logging
import os
import datetime
from shutil import copy
import glob
import re
import torch
import pickle
import shutil
import mlflow
from torch.utils.tensorboard import SummaryWriter


def set_logger():
    logging.basicConfig(level=logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s](%(filename)s:%(lineno)s) %(message)s")
    logger = logging.getLogger(__name__)
    logger.root.handlers[0].setFormatter(formatter)
    logger.info('start')
    return logger


class Logger():
    def __init__(self, args, description, use_tensorboard=True, use_mlflow=True, params=None):
        self.ignore_dirs = ['./__pycache__', './*log', './.idea', './runs', 'tmp', 'artifacts', 'mlruns']
        self.use_tensorboard = use_tensorboard
        self.use_mlflow = use_mlflow
        self.auto_end_run = False
        DIR = args.dir
        if not use_mlflow:
            os.makedirs(DIR, exist_ok=True)
            exp_num = 0
            while True:
                exp_dir = os.path.join(DIR, '{}'.format(exp_num))
                try:
                    os.makedirs(exp_dir, exist_ok=False)
                    break
                except:
                    exp_num += 1
            self.exp_dir = exp_dir
        else:
            if not mlflow.active_run():
                mlflow.start_run()
                self.auto_end_run = True
            path = mlflow.get_artifact_uri()
            if 'file://' in path:
                self.exp_dir = path[7:]  # remove file://
            else:
                self.exp_dir = path
            print(f'make mlflow logger: {self.exp_dir}')
        if self.use_tensorboard:
            print(f'make tensorboard logger: {self.exp_dir}')
            self.writer = SummaryWriter(self.exp_dir)
        self.log(args, description, params)

    def log(self, args, description, params=None):
        """log code"""
        code_dir = os.path.join(self.exp_dir, 'code')
        os.makedirs(code_dir, exist_ok=False)
        files = [p for p in glob.glob("./**/*.py", recursive=True) if os.path.isfile(p)]

        def check_ignore(input, ignore_dirs):
            match_flg = False
            for ignore_dir in ignore_dirs:
                match = re.findall(ignore_dir, input)
                if len(match) != 0:
                    match_flg = True
            return match_flg

        for f in files:
            if check_ignore(f, self.ignore_dirs):
                continue
            os.makedirs(os.path.dirname(os.path.join(code_dir, f)), exist_ok=True)
            if os.path.isfile(f):
                copy(f, os.path.join(code_dir, f))

        notename = os.path.join(self.exp_dir, 'log.txt')
        with open(notename, 'w') as note:
            content = ''

            """log descriptions"""
            now = datetime.datetime.now()
            content += (str(now) + '\n')
            content += (description + '\n')

            """log args"""
            for arg in vars(args):
                content += (arg + ' = ' + str(getattr(args, arg)) + '\n')

            """log params"""
            if params is not None:
                for param_name in params:
                    content += (param_name + ' = ' + str(params[param_name]) + '\n')
            note.write(content)

            """log params to tensorboard & mlflow"""
            if params is not None:
                self.add_params({**vars(args), **params})
            else:
                self.add_params(vars(args))

    def log_res(self, results, model=None, opt=None):
        """save model"""
        if model is not None:
            model_dir = os.path.join(self.exp_dir, 'model')
            os.makedirs(model_dir, exist_ok=False)
            if self.use_mlflow:
                for key in model:
                    model_path = os.path.join(model_dir, '{}'.format(key))
                    mlflow.pytorch.log_model(model[key].cpu(), 'model/{}'.format(key))
            else:
                for key in model:
                    model_path = os.path.join(model_dir, '{}.pt'.format(key))
                    torch.save(model[key].cpu().state_dict(), model_path)
        """save optimizer"""
        if opt is not None:
            opt_dir = os.path.join(self.exp_dir, 'opt')
            os.makedirs(opt_dir, exist_ok=True)
            for key in opt:
                model_path = os.path.join(opt_dir, '{}_opt.pt'.format(key))
                torch.save(opt[key].state_dict(), model_path)

        """log results"""
        if results is not None:
            data_dir = os.path.join(self.exp_dir, 'data')
            os.makedirs(data_dir, exist_ok=False)
            for res_name in results:
                res_path = os.path.join(data_dir, res_name)
                self.save(res_path, results[res_name])

        """finalize"""
        if self.use_tensorboard:
            self.writer.close()
        if self.auto_end_run:
            mlflow.end_run()

    def save(self, path, data):
        f = open(path, 'wb')
        pickle.dump(data, f)
        print('{} saved'.format(path))
        f.close()

    def add_params(self, params_dict):
        if self.use_mlflow:
            mlflow.log_params(params_dict)

    def add_scalar(self, *args):
        if self.use_tensorboard:
            self.writer.add_scalar(*args)
        if self.use_mlflow:
            mlflow.log_metric(*args)

    def add_figure(self, name, fig, global_step=None):
        if self.use_tensorboard:
            self.writer.add_figure(name, fig, global_step)

    def add_embedding(self, vecs, metadata):
        if self.use_tensorboard:
            self.writer.add_embedding(vecs, metadata=metadata)


