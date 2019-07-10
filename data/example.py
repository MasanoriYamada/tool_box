import mnist
import cifar10
import dsprites
import svhn
import lpc

def check_data(data_loader):
    train_iter = data_loader.get_train_loader()
    train_data_size = 0
    train_label_set= set()
    for data, label in train_iter:
        train_data_size += len(data)
        if label.numpy().shape[1:] == ():
            train_label_set = train_label_set.union(label.numpy())
        else:
            pass
        print('size:{} {}, type:{} {} min{}, max{}'.format(data.size(), label.size(), type(data), type(label), data.min(), data.max()))
    
    test_iter = data_loader.get_test_loader()
    test_data_size = 0
    test_label_set= set()
    for data, label in test_iter:
        test_data_size += len(data)
        if label.numpy().shape[1:] == ():
            test_label_set = test_label_set.union(label.numpy())
        else:
            pass
        print('size:{} {}, type:{} {} min{}, max{}'.format(data.size(), label.size(), type(data), type(label), data.min(), data.max()))
    
    print('train_size:{}, labels:{}'.format(train_data_size, train_label_set))
    print('test_size:{}, labels:{}'.format(test_data_size, test_label_set))



batch_size = 128

#data_loader = mnist.Loader(batch_size)
#check_data(data_loader)

#data_loader = cifar10.Loader(batch_size)
#check_data(data_loader)

#data_loader = dsprites.Loader(batch_size)
#check_data(data_loader)

#data_loader = svhn.Loader(batch_size)
#check_data(data_loader)

data_loader = lpc.Loader(batch_size)
check_data(data_loader)


