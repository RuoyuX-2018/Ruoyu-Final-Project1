import cv2
import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

def cat_img_depth():
    original_task_path = 'data/TP/COCOSearch18-images-TP/images'
    Depth_task_path = 'data/Depth/'
    Depth_list = os.listdir(Depth_task_path)
    for task in os.listdir(original_task_path):
        print('working on task ', task)
        original_img_path = original_task_path+'/'+task
        Depth_img_path = Depth_task_path+'/'+task
        for img_name in os.listdir(original_img_path):
            print(img_name)
            #read img
            img = cv2.imread(original_img_path+'/'+img_name)

            #read depth
            img_name_split = img_name.split('.')[0]
            depth = cv2.imread(Depth_img_path+'/'+img_name_split+'.jpg', cv2.IMREAD_GRAYSCALE)
            depth = np.expand_dims(depth, axis=2)

            #cat rgb and depth
            img = cv2.resize(img, (320, 512))
            rgbd = np.concatenate((img, depth), axis=2)
            np.save('data/RGBD/'+task+'/'+img_name_split+'.npy', rgbd)

def cat_dcb_depth():
    DCB_task_path = 'data/DCBs/LR/'
    Depth_task_path = 'data/Depth/'
    for task in os.listdir(Depth_task_path):
        print('working on task ', task)
        DCB_img_path = DCB_task_path + task
        Depth_img_path = Depth_task_path + task
        for depth_name in os.listdir(Depth_img_path):
            #read dcb
            depth_name_split = depth_name.split('.')[0]
            dcb = torch.load(DCB_img_path + '/' + depth_name_split + '.pth.tar')

            # read depth
            depth = Image.open(Depth_img_path + '/' + depth_name).convert('L')
            convert = transforms.ToTensor()
            depth = convert(depth.resize((32, 20)))

            # cat rgb and depth
            dcb_depth = torch.cat((dcb, depth), axis=0)
            torch.save(dcb_depth, 'data/DCB_Depth/LR/' + task + '/' + depth_name_split + '.pth.tar')


def cat_dcb_depth_multi():
    DCB_task_path = 'data/DCBs/LR/'
    Depth_task_path = 'data/Depth/'
    for task in os.listdir(Depth_task_path):
        print('working on task ', task)
        DCB_img_path = DCB_task_path + task
        Depth_img_path = Depth_task_path + task
        for depth_name in os.listdir(Depth_img_path):
            #read dcb
            depth_name_split = depth_name.split('.')[0]
            dcb = torch.load(DCB_img_path + '/' + depth_name_split + '.pth.tar')

            # read depth
            depth = Image.open(Depth_img_path + '/' + depth_name).convert('L')
            convert = transforms.ToTensor()
            depth = convert(depth.resize((32, 20)))

            # cat rgb and depth
            for i in range(dcb.shape[0]):
                dcb[i] = torch.multiply(dcb[i], depth)
            torch.save(dcb, 'data/DCB_Depth_multi/LR/' + task + '/' + depth_name_split + '.pth.tar')



if __name__ == '__main__':
    #cat_img_depth()
    #cat_dcb_depth()
    cat_dcb_depth_multi()

