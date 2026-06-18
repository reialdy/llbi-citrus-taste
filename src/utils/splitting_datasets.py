# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 01:08:55 2024
Splitting datasets program
@author: Reinaldy
"""

import os
import random
import shutil

data_path = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Blue\for_journal\Augmented\Manis"

# path to destination folders
train_folder = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Blue\for_journal\Splitted\Train\Manis"
val_folder = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Blue\for_journal\Splitted\Valid\Manis"
test_folder = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Blue\for_journal\Splitted\Test\Manis"

# Define a list of image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Create a list of image filenames in 'data_path'
imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]

# Sets the random seed 
random.seed(123)

# Shuffle the list of image filenames
random.shuffle(imgs_list)

# determine the number of images for each set
train_size = int(len(imgs_list) * 0.70)
val_size = int(len(imgs_list) * 0.20)
test_size = int(len(imgs_list) * 0.10)

# Create destination folders if they don't exist
# for folder_path in [train_folder, val_folder, test_folder]:
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

# Copy image files to destination folders
for i, f in enumerate(imgs_list):
    if i < train_size:
        dest_folder = train_folder
    elif i < train_size + val_size:
        dest_folder = val_folder
    else:
        dest_folder = test_folder
    shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))
