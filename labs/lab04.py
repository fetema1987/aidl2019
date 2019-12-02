# Set the random seed 123 from numpy
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html
import numpy as np
np.random.seed(123)
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms

# Let's define some hyper-parameters
hparams = {
    'batch_size':64,
    'num_epochs':10,
    'test_batch_size':64,
    'hidden_size':128,
    'num_classes':10,
    'num_inputs':784,
    'learning_rate':1e-3,
    'log_interval':100,
}

# we select to work on GPU if it is available in the machine, otherwise
# will run on CPU
hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# whenever we send something to the selected device (X.to(device)) we already use
# either CPU or CUDA (GPU). Importantly...
# The .to() operation is in-place for nn.Module's, so network.to(device) suffices
# The .to() operation is NOT in.place for tensors, so we must assign the result
# to some tensor, like: X = X.to(device)

mnist_trainset = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
mnist_testset = datasets.MNIST('data', train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

train_loader = torch.utils.data.DataLoader(
    mnist_trainset,
    batch_size=hparams['batch_size'],
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    mnist_testset,
    batch_size=hparams['test_batch_size'],
    shuffle=False)


    # We can retrieve a sample from the dataset by simply indexing it
img, label = mnist_trainset[0]
print('Img shape: ', img.shape)
print('Label: ', label)

# Similarly, we can sample a BATCH from the dataloader by running over its iterator
iter_ = iter(train_loader)
bimg, blabel = next(iter_)
print('Batch Img shape: ', bimg.shape)
print('Batch Label shape: ', blabel.shape)
print('The Batched tensors return a collection of {} grayscale images ({} channel, {} height pixels, {} width pixels)'.format(bimg.shape[0],
                                                                                                                              bimg.shape[1],
                                                                                                                              bimg.shape[2],
                                                                                                                              bimg.shape[3]))
print('In the case of the labels, we obtain {} batched integers, one per image'.format(blabel.shape[0]))


# Definition of a Python function that plots a NxN grid of images contained
# in the images array
def plot_samples(images,N=5):

    '''
    Plots N**2 randomly selected images from training data in a NxN grid
    '''

    # Randomly select NxN images and save them in ps
    ps = random.sample(range(0,images.shape[0]), N**2)

    # Allocates figure f divided in subplots contained in an NxN axarr
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html
    f, axarr = plt.subplots(N, N)

    # Index for the images in ps to be plotted
    p = 0

    # Scan the NxN positions of the grid
    for i in range(N):
        for j in range(N):

            # Load the image pointed by p
            im = images[ps[p]]

            # If images are encoded in grayscale
            # (a tensor of 3 dimensions: width x height x luma)...
            if len(images.shape) == 3:
              # ...specify the colormap as grayscale
              # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.imshow.html
              axarr[i,j].imshow(im,cmap='gray')
            else:
              # ...no need to specify any color map
              axarr[i,j].imshow(im)

            # Remove axis
            axarr[i,j].axis('off')

            # Point to the next image from the random selection
            p+=1
    # Show the plotted figure
    plt.show()

# convert the dataloader output tensors from the previous cell to numpy arrays
# The channel dimension has to be squeezed in order for matplotlib to work
# with grayscale images
img = bimg.squeeze(1).data.numpy()
plot_samples(img)
