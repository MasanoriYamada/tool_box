import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import base64
import time
from selenium import webdriver
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle



class LPC(Dataset):
    def __init__(self, data_range, download=True):
        self.data_path = './data/lpc/lpc.pickle'
        if download:
            self.download()
        with open(self.data_path, 'rb') as f:
            self.data, self.label = pickle.load(f)
        self.data = self.change_range(self.data, data_range[0], data_range[1])
        self.data = self.to_tensor(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], idx
    
    def to_tensor(self, x):
        # x[batch, seq, c, h, w]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return x

    def change_range(self, data, min_v, max_v):
        n_data = data.reshape(-1, 1)
        n_data = n_data * (max_v - min_v) + min_v
        n_data = n_data.reshape(data.shape)
        return n_data
    
    def download(self):
        dir_path = os.path.dirname(self.data_path)
        os.makedirs(dir_path, exist_ok=True)

        if not os.path.exists(self.data_path):
            print('download lpc dataset now...')
            self.lpc_download()

    def lpc_download(self):
        def prepare_tensor(path):
            img = Image.open(path)
            img = img.convert("RGB")
            img = np.array(img)

            actions = {
                'walk': {
                    'range': [(9, 10), (10, 11), (11, 12)],  # row
                    'frames': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]  # col
                },
                'spellcast': {
                    'range': [(1, 2), (2, 3), (3, 4)],
                    'frames': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (6, 7)]
                },
                'slash': {
                    'range': [(13, 14), (14, 15), (15, 16)],
                    'frames': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 6), (5, 6)]
                }
            }
        
            slices = []
            slice_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            for action, params in actions.items():
                for row in params['range']:
                    sprite = []
                    for col in params['frames']:
                        sprite.append(slice_transform(img[64 * row[0]:64 * row[1], 64 * col[0]:64 * col[1], :]))
                    slices.append(torch.stack(sprite))
            return slices
    
        driver = webdriver.Firefox()
        driver.get("http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator/")
        driver.maximize_window()
        '''
        bodies = ['light', 'dark', 'dark2', 'darkelf', 'darkelf2', 'tanned', 'tanned2']
        shirts = ['longsleeve_brown', 'longsleeve_teal', 'longsleeve_maroon', 'longsleeve_white']
        hairstyles = ['green', 'blue', 'pink', 'raven', 'white', 'dark_blonde']
        pants = ['magenta', 'red', 'teal', 'white', 'robe_skirt']
        '''
        bodies = ['light', 'dark2']
        shirts = ['longsleeve_brown', 'longsleeve_teal']
        hairstyles = ['green', 'pink']
        pants = ['red', 'teal']
        train = 0
        test = 0
        states = []
        ids = []
        for id0, body in enumerate(bodies):
            driver.execute_script("return arguments[0].click();", driver.find_element_by_id('body-' + body))
            time.sleep(0.5)
            for id1, shirt in enumerate(shirts):
                driver.execute_script("return arguments[0].click();", driver.find_element_by_id('clothes-' + shirt))
                time.sleep(0.5)
                for id2, pant in enumerate(pants):
                    if pant == 'robe_skirt':
                        driver.execute_script("return arguments[0].click();",
                                              driver.find_element_by_id('legs-' + pant))
                    else:
                        driver.execute_script("return arguments[0].click();",
                                              driver.find_element_by_id('legs-pants_' + pant))
                    time.sleep(0.5)
                    for id3, hair in enumerate(hairstyles):
                        driver.execute_script("return arguments[0].click();",
                                              driver.find_element_by_id('hair-plain_' + hair))
                        time.sleep(0.5)
                        name = body + "_" + shirt + "_" + pant + "_" + hair
                        print("Creating character: " + "'" + name)
                        canvas = driver.find_element_by_id('spritesheet')  #
                        canvas_base64 = driver.execute_script(
                            "return arguments[0].toDataURL('image/png').substring(21);",
                            canvas)
                        canvas_png = base64.b64decode(canvas_base64)
                        tmp_path = 'data/lpc/' + str(name) + '.png'
                        with open(tmp_path, "wb") as f:
                            f.write(canvas_png)
                        slices = prepare_tensor(tmp_path)
                        print("Dimension is {}".format(slices[0].shape))
                        p = torch.rand(
                            1).item() <= 0.1  # Randomly add 10% of the characters created in the test set
                    
                        id_ = [id0, id1, id2, id3]
                        # data[0] = [batch, seq, dim] state
                        # data[1]= [batch, id_] action
                        # slice = [batch,seq,channel,xdim,ydim]
                        for state in slices:
                            states.append((state.numpy() + 1.0) / 2.0)
                            ids.append(id_)
        states = np.asarray(states)
        ids = np.asarray(ids)
        with open(self.data_path, 'wb') as f:
            pickle.dump([states, ids], f)
        # data[0].shape = [batch, seq, dim] state
        # data[1].shape= [batch, id_] action
        print("Dataset is Ready.Training Set Size : %d. Test Set Size : %d " % (train, test))


class Loader():
    def __init__(self, batch_size):
        self.data_range = [-1, 1]
        self.train_dataset = LPC(self.data_range)
        self.data_len = len(self.train_dataset)
        self.seq_len = 8
        self.f_dim = 3 * 64 * 64
        self.data_shape = (self.data_len, self.seq_len, self.f_dim)
        self.cnn_shape = (3, 64, 64)
        self.label_num = [2, 2, 2, 2, 3, 3]
        self.num_factors = len(self.label_num)
        self.factor = ['skin', 'shirts', 'hair', 'pants', 'direction', 'motion']
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=batch_size, shuffle=True, **kwargs)
    
    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader
    
    def sample(self, num, random_state):
        # for calc metric
        idx = random_state.randint(self.data_len, size=num)
        x, f = self.train_dataset[idx]
        return f, x
    
    def sample_factors(self, num, random_state):
        # for beta_vae metric
        idx = random_state.randint(self.data_len, size=num)
        x, f = self.train_dataset[idx]
        return f
    
    def sample_observations_from_factors(self, factors, random_state):
        # for beta_vae metric
        all_data, all_factor = self.train_dataset[:]
        ret = []
        for target_factor in factors:
            idx = np.all(all_factor == target_factor, axis=1)
            data_cond_label = all_data.numpy()[idx]
            id = random_state.randint(len(data_cond_label))
            ret.append(data_cond_label[id])
        ret = torch.from_numpy(np.asarray(ret, dtype=np.float32))
        return ret

