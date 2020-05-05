import numpy as np
import torch
print(torch.__version__)
import torch.nn as NN
import torch.nn.functional as F

import torchvision
print(torchvision.__version__)


import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Problem1 import *
from Problem2 import *




# print(cv2.__version__)
print('Import Initializations complete')



flag = False
prgRun = True

def main(prgRun):
    if __name__ == '__main__':
        problem = 2
        problem = int(input('Enter 1 for MNIST \nEnter 2 for CFIR \nWhich part would you like to run: '))

        if problem==1:
            Problem1Main()
        elif problem==2:
            Problem2Main()
        else:
            prgRun = False
            return prgRun



        prgRun = False
        return prgRun
print('Function Initializations complete')

if __name__ == '__main__':
    print('Start Program')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()