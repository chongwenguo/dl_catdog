'''
Load data
'''

import os
import traceback
import cv2
import numpy as np
from PIL import Image
from shutil import copyfile
from collections import defaultdict
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class MyDataset(data.Dataset):

    def __init__(self, path_file, list_file,numJoints,type):
        pass
#
#         Args:
#           path_file: (str) heatmap and optical file location
#           list_file: (str) path to index file.
#           numJoints: (int) number of joints
#           type: (boolean) use pose flow(true) or optical flow(false)
#
#
#         self.numJoints = numJoints
#
#         # read heatmap and optical path
#         with open(path_file) as f:
#             paths = f.readlines()
#
#         for path in paths:
#             splited = path.strip().split()
#             if splited[0]=='resPath':
#                 self.resPath = splited[1]
#             elif splited[0]=='gtPath':
#                 self.gtPath = splited[1]
#             elif splited[0]=='opticalFlowPath':
#                 self.opticalFlowPath = splited[1]
#             elif splited[0]=='poseFlowPath':
#                 self.poseFlowPath = splited[1]
#         if type:
#             self.flowPath = self.poseFlowPath
#         else:
#             self.flowPath = self.opticalFlowPath
#
#
#         #read list
#         with open(list_file) as f:
#             self.list = f.readlines()
#             self.num_samples = len(self.list)
#
# def __getitem__(self, idx):
#     '''
#     load heatmaps and optical flow and encode it to a 22 channels input and 6 channels output
#     :param idx: (int) image index
#     :return:
#         input: a 22 channel input which integrate 2 optical flow and heatmaps of 3 image
#         output: the ground truth
#     '''
#     input = []
#     output = []
#     # load heatmaps of 3 image
#     for im in range(3):
#         for map in range(6):
#             curResPath = self.resPath + self.list[idx].rstrip('\n') + str(im + 1) + '/' + str(map + 1) + '.bmp'
#             heatmap = Image.open(curResPath)
#             heatmap.load()
#             heatmap = np.asarray(heatmap, dtype='float') / 255
#             input.append(heatmap)
#     # load 2 flow
#     for flow in range(2):
#         curFlowXPath = self.flowPath + self.list[idx].rstrip('\n') + 'flowx/' + str(flow + 1) + '.jpg'
#         flowX = Image.open(curFlowXPath)
#         flowX.load()
#         flowX = np.asarray(flowX, dtype='float')
#         curFlowYPath = self.flowPath + self.list[idx].rstrip('\n') + 'flowy/' + str(flow + 1) + '.jpg'
#         flowY = Image.open(curFlowYPath)
#         flowY.load()
#         flowY = np.asarray(flowY, dtype='float')
#         input.append(flowX)
#         input.append(flowY)
#     # load groundtruth
#     for map in range(6):
#         curgtPath = self.resPath + self.list[idx].rstrip('\n') + str(2) + '/' + str(map + 1) + '.bmp'
#         heatmap = Image.open(curResPath)
#         heatmap.load()
#         heatmap = np.asarray(heatmap, dtype='float') / 255
#         output.append(heatmap)
#
#     input = torch.Tensor(input)
#     output = torch.Tensor(output)
#
#     return input,output


if __name__ == '__main__':
    cur_dir = os.path.dirname(__file__)
    file_name_path = os.path.join(cur_dir, 'data/images/')
    i = 0
    trainval = open('data/annotations/trainval.txt', "r")
    test = open('data/annotations/test.txt', "r")
    for line in trainval:
        if '#' in line:
            continue
        l = line.split()
        image_name = l[0] + ".jpg"
        breed_name = l[0][0: l[0].rfind('_')]
        class_id = l[1]
        breed = l[3]
        if l[2] == '1':
            species = 'cat'
        else:
            species = 'dog'
        if not os.path.exists("data/trainval/species/cat/"):
                os.mkdir("data/trainval/species/cat/")
        if not os.path.exists("data/trainval/species/dog/"):
                os.mkdir("data/trainval/species/dog/")
        copyfile(file_name_path + image_name, "data/trainval/species/"+ species + '/' + image_name)
        if species == 'cat':
            if not os.path.exists("data/trainval/breeds_cat/"+ breed_name):
                os.mkdir("data/trainval/breeds_cat/"+ breed_name)
            copyfile(file_name_path + image_name, "data/trainval/breeds_cat/"+ breed_name + '/' + image_name)
        if species == 'dog':
            if not os.path.exists("data/trainval/breeds_dog/" + breed_name):
                os.mkdir("data/trainval/breeds_cat/" + breed_name)
            copyfile(file_name_path + image_name, "data/trainval/breeds_dog/" + breed_name + '/' + image_name)

    for line in test:
        if '#' in line:
            continue
        l = line.split()
        image_name = l[0] + ".jpg"
        breed_name = l[0][0: l[0].rfind('_')]
        class_id = l[1]
        breed = l[3]
        if l[2] == '1':
            species = 'cat'
        else:
            species = 'dog'

        if not os.path.exists("data/test/species/cat/"):
                os.mkdir("data/test/species/cat/")
        if not os.path.exists("data/test/species/dog/"):
                os.mkdir("data/test/species/dog/")
        copyfile(file_name_path + image_name, "data/test/species/"+ species + '/' + image_name)
        if species == 'cat':
            if not os.path.exists("data/test/breeds_cat/"+ breed_name):
                os.mkdir("data/test/breeds_cat/"+ breed_name)
            copyfile(file_name_path + image_name, "data/trainval/breeds_cat/"+ breed_name + '/' + image_name)
        if species == 'dog':
            if not os.path.exists("data/test/breeds_dog/" + breed_name):
                os.mkdir("data/test/breeds_dog/" + breed_name)
            copyfile(file_name_path + image_name, "data/test/breeds_dog/" + breed_name + '/' + image_name)


 # map = {"Abyssinian":197,
    #     "american_bulldog":199,
    #     "american_pit_bull_terrier":199,
    #     "basset_hound":199,
    #     "beagle":199,
    #     "Bengal":199,
    #     "Birman":199,
    #     "Bombay":183,
    #     "boxer":198,
    #     "British_Shorthair":199,
    #     "chihuahua":199,
    #     "Egyptian_Mau":189,
    #     "english_cocker_spaniel":195,
    #     "english_setter":199,
    #     "german_shorthaired":199,
    #     "great_pyrenees":199,
    #     "havanese":199,
    #     "japanese_chin":199,
    #     "keeshond":198,
    #     "leonberger":199,
    #     "Maine_Coon":199,
    #     "miniature_pinscher":199,
    #     "newfoundland":195,
    #     "Persian":199,
    #     "pomeranian":199,
    #     "pug":199,
    #     "Ragdoll":199,
    #     "Russian_Blue":199,
    #     "saint_bernard":199,
    #     "samoyed":199,
    #     "scottish_terrier":198,
    #     "shiba_inu":199,
    #     "Siamese":198,
    #     "Sphynx":199,
    #     "staffordshire_bull_terrier":188,
    #     "wheaten_terrier":199,
    #     "yorkshire_terrier":199
    #     }