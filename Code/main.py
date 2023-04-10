# Import required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import copy
# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg") / 255
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask

def findSeam(energy):
    #Finds the minimum seam
    energyCpy = copy.deepcopy(energy)
    backtrack = np.zeros(energy.shape)
    for x in range(0, energy.shape[0]-1):
        for y in range(0, energy.shape[-1]-1):
            minE = 0
            if y ==0:
                idx = np.argmin(energyCpy[x-1, y:y+2])
                backtrack[x, y] = idx + y
                minE = energyCpy[x-1, idx+y]
            else:
                idx = np.argmin(energyCpy[x-1, y-1:y+2])
                backtrack[x, y] = idx + y -1
                minE = energyCpy[x-1, idx+y-1]

            energyCpy[x, y]+= minE
    return energyCpy, backtrack

def removeSeam(energy):
    #Returns the index for which to remove seams
    energyCpy, backtrack = findSeam(energy)
    msk = np.ones(energyCpy.shape, dtype=int)
    y = np.argmin(energyCpy[-1])
    for x in reversed(range(energyCpy.shape[0])):
        msk[x, y] = 0
        y = int(backtrack[x, y])
    return msk

def SeamCarve(input, widthFac, heightFac, mask):

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    #Add a case here to rotate it by 90 degrees to for width

    inSize = input.shape
    size   = (int(widthFac*inSize[1]), int(heightFac*inSize[0]))

    #1.) Computing the energy function
    grayIn = input
    if len(input.shape)>2: #Convert Image to grayscale if necessary
        grayIn = np.sum(input, axis=-1)/input.shape[-1]
    
    gradX = np.diff(grayIn, axis=-1)
    gradY = np.diff(grayIn, axis=0)
    #The gradient shapes being incorrect is not really a problem,
    #As we dont want to be carving along the edges
    padX = np.pad(gradX, [[0,0], [0,1]])
    padY = np.pad(gradY, [[0,1], [0,0]])
    energy = padX + padY
    mask = removeSeam(energy)

    #Add a case here to unrotate it by 90 degrees if width

    #Need to make the mask 3 dimensional
    mask = np.stack([mask]*3, axis=2)
    print(input.shape, input[mask].shape, size)
    return input[mask].reshape([input.shape[0], input.shape[1]-1, 3])
    #return cv2.resize(input[mask].reshape(input.shape[0], input.shape[1]-1, 3), size), size


# Setting up the input output paths
inputDir = '../Images/'
outputDir = '../Results/'

widthFac = 0.5; # To reduce the width, set this parameter to a value less than 1
heightFac = 1;  # To reduce the height, set this parameter to a value less than 1
N = 4 # number of images

for index in range(1, N + 1):

    input, mask = Read(str(index).zfill(2), inputDir)

    # Performing seam carving. This is the part that you have to implement.
    output, size = SeamCarve(input, widthFac, heightFac, mask)

    # Writing the result
    plt.imsave("{}/result_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[0]).zfill(2), 
                                            str(size[1]).zfill(2)), output)