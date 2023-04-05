# Import required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

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

def SeamCarve(input, widthFac, heightFac, mask):

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    inSize = input.shape
    size   = (int(widthFac*inSize[1]), int(heightFac*inSize[0]))

    #1.) Computing the energy function
    grayIn = input
    if len(input.shape)>2: #Convert Image to grayscale if necessary
        grayIn = np.sum(input, axis=-1)/input.shape[-1]
    
    gradX = np.diff(grayIn, axis=-1)
    gradY = np.diff(grayIn, axis=0)

    return cv2.resize(input, size), size


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