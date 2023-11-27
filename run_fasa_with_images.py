#!/usr/bin/env python

# @author: Sardor Abdirayimov 
# @inspired by https://mpatacchiola.github.io/blog/ and
# https://github.com/mpatacchiola/deepgaze/tree/master

import numpy as np
import cv2
from timeit import default_timer as timer
from fasa import FasaSaliencyMapping
import os



def main():
    image_files = [f for f in os.listdir("./images/")]
    print("List of image files: ", image_files)
    image_salients, original_images = [], []
    # for each image the same operations are repeated
    for image in image_files:
        image = cv2.imread("./images/" + image)
        print(image.shape)
        my_map = FasaSaliencyMapping(image.shape[0], image.shape[1])
        start = timer()
        image_salient = my_map.returnMask(image, tot_bin=8, format="BGR2LAB")
        image_salient = cv2.GaussianBlur(image_salient, (3,3), 1) # to make image pretty
        end = timer()
        image_salients.append(image_salient)
        original_images.append(image)
        print("--- %s One image total seconds ---" % (end - start))

    # Creating stack of images and showing them on screen 
    original_images_stack = np.hstack(original_images)
    saliency_images_stack = np.hstack(image_salients)
    saliency_images_stack = np.dstack((saliency_images_stack, saliency_images_stack, saliency_images_stack))
    cv2.imshow("Original-Saliency", np.vstack((original_images_stack, saliency_images_stack)))
    cv2.imwrite("results.png", np.vstack((original_images_stack, saliency_images_stack)))
    while True:
        if cv2.waitKey(33) == ord("q"):
            cv2.destroyAllWindows()
            break
    

if __name__ == "__main__":
    main()