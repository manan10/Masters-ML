# -*- coding: utf-8 -*-
from skimage import io, img_as_float
import sys
import numpy as np
import os

def kMeansClustering(imageVectors, k, numOfIters):
    lbs = np.full((imageVectors.shape[0],), -1)
    clstr_proto = np.random.rand(k, 3)
    
    for i in range(numOfIters):
        print('Iteration: ' + str(i + 1))
        points_label = [None for k_i in range(k)]

        for rgb_i, rgb in enumerate(imageVectors):          
            rgb_row = np.repeat(rgb, k).reshape(3, k).T
            closest_label = np.argmin(np.linalg.norm(rgb_row - clstr_proto, axis=1))
            lbs[rgb_i] = closest_label

            if (points_label[closest_label] is None):
                points_label[closest_label] = []

            points_label[closest_label].append(rgb)
       
        for k_i in range(k):
            if (points_label[k_i] is not None):
                new_cluster_prototype = np.asarray(points_label[k_i]).sum(axis=0) / len(points_label[k_i])
                clstr_proto[k_i] = new_cluster_prototype
                
    return (lbs, clstr_proto)
    
if len(sys.argv) == 4:
    imageFile = sys.argv[1]
    kVal = int(sys.argv[2])
    noOfIters = int(sys.argv[3])
    outputImageName = imageFile.split('.')[0] + '-50-k=' + str(kVal) + '.' + imageFile.split('.')[1]
else:
    sys.exit("\nIncorrect Arguments\nPlease provide following arguments: \n \t- Path to Image File \n\t- Value Of K \n\t- Number of Iterations")

image = io.imread(imageFile)[:, :, :3]
image = img_as_float(image)

imageDimentions = image.shape
imageVectors = image.reshape(-1, image.shape[-1])

lbs, color_centroids = kMeansClustering(imageVectors, kVal, noOfIters)

output_image = np.zeros(imageVectors.shape)
for i in range(output_image.shape[0]):
    output_image[i] = color_centroids[lbs[i]]

output_image = output_image.reshape(imageDimentions)
print('Saving the Compressed Image')
io.imsave(outputImageName , output_image)
print('Image Compression Completed')
info = os.stat(imageFile)
print("Image size before : ", info.st_size/1024,"KB")
info = os.stat(outputImageName)
print("Image size : ", info.st_size/1024,"KB")

