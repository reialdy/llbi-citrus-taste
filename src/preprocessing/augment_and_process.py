# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 01:31:25 2024
@author: Reinaldy
LLBI Image Pre-Processing Program 
"""

import cv2
import os
import numpy as np
import imutils

# define the location of the folder to go through
directory = "D:\\College\\Skripsi\\Image Acquisition\\Jeruk Siam Garut\\LLBI\\Labeled - 2 Class\\Blue\\for_journal\\Asam"
file_list = os.listdir(directory)
file_list

# Image Processing + Augmented Data
 for file in file_list:
         if file.endswith(".jpeg"):
             img_path = os.path.join(directory, file)
             image = cv2.imread(img_path)
             for angle in np.arange(0, 360, 5):
                 img_rotated = imutils.rotate_bound(image, angle)
                 img_resized = cv2.resize(img_rotated, (256,256))
                 # img_rotated = imutils.rotate_bound(img_resized, angle)
                 # #-------------------- Digital image processing --------------------#
                 img = os.path.join(directory, file)
                 image = cv2.imread(img)
                 # -------------------- Digital image processing --------------------#
        
                 # Convert from BGR to Grayscale
                 # image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
                 lower_bound = 150
                 upper_bound = 255
                 mask = cv2.inRange(image_gray, lower_bound, upper_bound)
                 img_final = mask

                 # Get the file name without the extension
                 filename, file_extension = os.path.splitext(file)

                 # Create the final file name with the rotation degree
                 final_filename = f"processed_{filename}_{angle}deg{file_extension}"

                 cv2.imwrite(os.path.join("D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Red\Processed Mask - Undersampling\Manis", final_filename), img_final)
