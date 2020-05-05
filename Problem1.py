from __main__ import *
import torchvision.transforms as transforms

from MNistCNN import *
from MNisttrainNN import *


# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


def Problem1Main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    classes = ('0', '1', '2', '3', '4', '5', '5', '7', '8', '9')

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=True, download=True,
                                                                          transform=transforms.Compose(
                                                                              [transforms.ToTensor(),
                                                                               transforms.Normalize((0.1307,),
                                                                                                    (0.3081,))])),
                                               batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=False,
                                                                         transform=transforms.Compose(
                                                                             [transforms.ToTensor(),
                                                                              transforms.Normalize((0.1307,),
                                                                                                   (0.3081,))])),
                                              batch_size=64, shuffle=True)

    CNN = MNistCNN()
    MNisttrainNN(CNN, batch_size=64, epochs=5, lr=.01, train_loader=train_loader, test_loader=test_loader,
                 classes=classes)
