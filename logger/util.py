import logging
import os
import datetime
from shutil import copy
import glob
import re
import torch
import pickle


def set_logger():
    logging.basicConfig(level=logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s](%(filename)s:%(lineno)s) %(message)s")
    logger = logging.getLogger(__name__)
    logger.root.handlers[0].setFormatter(formatter)
    logger.info('start')
    return logger


class Logger():
    def __init__(self, args, description, params=None):
        self.ignore_dirs = ['./__pycache__', './*log', './.idea', './runs', 'tmp']
        DIR = args.dir
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
            if not params == None:
                for param_name in params:
                    content += (param_name + ' = ' + str(params[param_name]) + '\n')
            note.write(content)
    
    def log_res(self, results, model=None, opt=None):
        """save model"""
        if model is not None:
            model_dir = os.path.join(self.exp_dir, 'model')
            os.makedirs(model_dir, exist_ok=False)
            for key in model:
                model_path = os.path.join(model_dir, '{}.pt'.format(key))
                torch.save(model[key].cpu().state_dict(), model_path)
        
        """save optimizer"""
        if opt is not None:
            opt_dir = os.path.join(self.exp_dir, 'model')
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
    
    def save(self, path, data):
        f = open(path, 'wb')
        pickle.dump(data, f)
        print('{} saved'.format(path))
        f.close()
