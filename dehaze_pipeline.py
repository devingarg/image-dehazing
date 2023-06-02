import os
import cv2
import numpy as np
import heapq

def findAtmophericLight(img, darkChannel):
    # get the top 0.1% pixels from the dark channel, maintain a max heap to do that
    h = []
    for i in range(darkChannel.shape[0]):
        for j in range(darkChannel.shape[1]):
            heapq.heappush(h, (darkChannel[i, j].item(), (i, j)))


    HOW_MANY = int(0.1/100 * len(img.flatten()))

    top = heapq.nlargest(HOW_MANY, h)

    topPix = []
    for _, pix in top:
        topPix.append(pix)

    # iterate over all the pixels to find the highest intensity values
    A = np.zeros(3)
    for px, py in topPix:
        A += img[px,py]
    A = A/len(topPix)

    return A

def recoverImage(img, t, A=np.array([1,1,1])):
    t_0 = 0.1
    J = np.zeros(img.shape, dtype='float64')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            J[i,j] = ((img[i,j] - A) / max(t[i,j], t_0)) + A
            
    return J

def haze2dehazed(haze_dir, transmission_map_dir, output_dir):
    haze_path = os.listdir(haze_dir)
    tm_path = os.listdir(transmission_map_dir)

    for image_name, transmission_map_name in zip(haze_path, tm_path):

        img_full_path = os.path.join(haze_dir, image_name)
        t_full_path = os.path.join(transmission_map_dir, transmission_map_name)

        img = cv2.imread(img_full_path)/255
        t = cv2.imread(t_full_path)/255

        # tmap should be 1-channel image
        t = cv2.cvtColor(t.astype('float32'), cv2.COLOR_BGR2GRAY)

        # recover image using transmission map
        dehaze_img = recoverImage(img, t)

        # write to disk
        dehaze_img_name = os.path.join(output_dir, image_name.replace('syn','rst'))
        cv2.imwrite(dehaze_img_name, dehaze_img*255)

