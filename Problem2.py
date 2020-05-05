from __main__ import *
import torchvision.transforms as transforms

from CNNBuild import *
from trainNN import *



# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


def Problem2Main():

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    classes = ('0 plane', '1 car', '2 bird', '3 cat', '4 deer', '5 dog',
               '6 frog', '7 horse', '8 ship', '9 truck')

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('./cifardata', train=True, download=True,
                                                                            transform=transforms.Compose(
                                                                                [transforms.ToTensor(),
                                                                                 transforms.Normalize((0.5,),
                                                                                                      (0.5,))])),
                                               batch_size=128, shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('./cifardata', train=False,
                                                                           transform=transforms.Compose(
                                                                               [transforms.ToTensor(),
                                                                                transforms.Normalize((0.5,), (0.5,))])),
                                              batch_size=64, shuffle=True)


    CNN = CNNBuild()
    trainNN(CNN, batch_size=128, epochs=36, lr=.1, train_loader=train_loader, test_loader=test_loader, classes=classes)
