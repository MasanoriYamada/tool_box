import torch.utils.data
from torchvision import datasets, transforms


class Loader():
    def __init__(self, batch_size, alpha=1.0, beta=0.0):
        self.fc_shape = (-1, 28 * 28)
        self.cnn_shape = (-1, 1, 28, 28)
        self.label_num = [10]
        self.factor = ['number']
        self.label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.range = [-1,1]
        self.train_len = 60000
        self.test_len = 10000
        self.name = 'mnist'
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.test_batch_size = batch_size

        data_transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ScaleShift(self.alpha, self.beta)])
        data_transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ScaleShift(self.alpha, self.beta)])
        train_path = 'data/mnist/'
        test_path = 'data/mnist/'
        train_dataset = datasets.MNIST(train_path, train=True, download=True, transform=data_transform_train)
        test_dataset = datasets.MNIST(test_path, train=False, transform=data_transform_test)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True, num_workers=2)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader


class ScaleShift(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        if self.alpha == 1.0 and self.beta == 0.0:
            return sample
        else:
            return sample * self.alpha + self.beta
