from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch


class DataFrame(object):
    def __init__(self, df_x, df_y, transform):
        assert len(df_x) == len(df_y), 'Err len(df_x):{} != len(df_y):{}'.format(len(df_x), len(df_y))
        self.x = torch.from_numpy(df_x.values)
        self.y = torch.from_numpy(df_y.values)
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.transform is not None:
            self.x[index] = self.transform(self.x[index])
        return self.x[index], self.y[index]


class Loader():
    def __init__(self, df_x, df_y, batch_size, train_size=0.5, transform=None):
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=train_size, shuffle=False)
        self.train_dataset = DataFrame(x_train, y_train, transform)
        self.test_dataset = DataFrame(x_test, y_test, transform)
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=batch_size, shuffle=False, **kwargs)
        self.fc_shape = (-1, len(df_x.columns))
        self.x_label = list(df_x.columns)
        self.y_label = list(df_y.columns)
        self.label_num = df_y.nunique().values
        self.train_len = len(x_train)
        self.test_len = len(x_test)
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
