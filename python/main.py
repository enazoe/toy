#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

root_dir = './data'
file_name = "train.txt"
img_suffix = 'jpg'

def get_all_imgs_path(root_dir,imgs):
    files = os.listdir(root_dir)
    for file in files:
        file_path = os.path.join(root_dir,file)
        if os.path.isdir(file_path):
            get_all_imgs_path(file_path,imgs)
        elif file.split('.')[-1]=='txt':
            imgs.append(os.path.join(file_path))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    imgs = []
    get_all_imgs_path(root_dir,imgs)
    imgs = [file.replace('txt',img_suffix) for file in imgs]
    with open(os.path.join(file_name),'w') as file:
        for line in imgs:
            file.write(line+'\n')
    print(imgs)
