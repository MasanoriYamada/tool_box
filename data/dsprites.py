import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from numpy import prod
import shutil
import requests


class Shapes(object):
    def __init__(self, shape_form, download=True):
        self.data_path = 'data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.label_num = [1, 3, 6, 40, 32, 32]
        self.factor = ['color','shape','scale','orientation','posX','posY']
        if download:
            self.download()
        self.dataset_zip = np.load(self.data_path, encoding='latin1')
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()
        self.labels = torch.from_numpy(self.dataset_zip['latents_classes']).float()

        id_list = []  # for selecting labels
        idx = []
        label_num = []
        factor = []
        for id, (key, n_factor) in enumerate(zip(self.factor, self.label_num)):
            cut_type = shape_form[key]
            if cut_type == 'fix':
                idx.append(n_factor//2)
            elif cut_type == 'full':
                label_num.append(n_factor)
                idx.append(slice(None))
                factor.append(key)
                id_list.append(id)
            elif cut_type == 'half':
                label_num.append(n_factor//2)
                idx.append(slice(None,None,2))
                factor.append(key)
                id_list.append(id)
            else:
                print('ERR: dsprites dataset is worng')
        imgs_idx = tuple(idx + [slice(None), slice(None)])
        idx = tuple(idx)
        self.factor = factor
        self.imgs = self.imgs.reshape(1, 3, 6, 40, 32, 32, 1, 64, 64)
        self.imgs = self.imgs[imgs_idx].reshape(-1, 1, 64, 64)
        self.labels = self.labels.reshape(1, 3, 6, 40, 32, 32, 6)
        self.labels = self.labels[idx].reshape(-1, 6)[:,id_list]
        self.label_num = label_num
        
    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        t = self.labels[index]
        return x, t

    def download(self):
        url = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        dir_path = os.path.dirname(self.data_path)
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(self.data_path):
            print('download dsprites dataset now...')
            res = requests.get(url, stream=True)
            with open(self.data_path, 'wb') as fp:
                shutil.copyfileobj(res.raw, fp)




class Loader():
    def __init__(self, batch_size, data_type=None):
        if data_type is None:
            shape_form = {
            'color':'full',
            'shape':'full',
            'scale':'full',
            'orientation':'full',
            'posX':'full',
            'posY':'full'
            }
        elif data_type == 'scale_posX_posY':
            shape_form = {
            'color':'fix',
            'shape':'fix',
            'scale':'full',
            'orientation':'fix',
            'posX':'full',
            'posY':'full'
            }
        elif data_type == 'full_posX':
            shape_form = {
            'color':'fix',
            'shape':'fix',
            'scale':'fix',
            'orientation':'fix',
            'posX':'full',
            'posY':'fix'
            }
        elif data_type == 'half_posX':
            shape_form = {
            'color':'fix',
            'shape':'fix',
            'scale':'fix',
            'orientation':'fix',
            'posX':'half',
            'posY':'fix'
            }
        elif data_type == 'posY_rot':
            shape_form = {
            'color':'fix',
            'shape':'fix',
            'scale':'fix',
            'orientation':'full',
            'posX':'fix',
            'posY':'full'
            }
        elif data_type == 'rot':
            shape_form = {
            'color':'fix',
            'shape':'fix',
            'scale':'fix',
            'orientation':'full',
            'posX':'fix',
            'posY':'fix'
            }
        elif data_type == 'test':
            shape_form = {
            'color':'fix',
            'shape':'fix',
            'scale':'full',
            'orientation':'fix',
            'posX':'half',
            'posY':'half'
            }
        
        self.train_dataset = Shapes(shape_form)
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=batch_size, shuffle=False, **kwargs)
        self.fc_shape = (-1, 64 * 64)
        self.cnn_shape = (-1, 1, 64, 64)
        self.range = [0,1]
        self.factor = self.train_dataset.factor
        self.label_num =self.train_dataset.label_num
        self.len = prod(self.label_num)
        self.name = 'dsprites'

    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
