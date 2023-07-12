from __future__ import print_function, absolute_import

import random
import torch.utils.data as data
# from pose.utils.osutils import *
# from pose.utils.transforms import *

from scipy.io import loadmat
import argparse
from .keypoint_dataset import Animal14KeypointDataset
from .util import isfile
import os
import numpy as np
import torch
import json
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from .util import isfile, im_to_torch, fliplr, shufflelr_ori, crop_ori, color_normalize, to_torch, transform, draw_labelmap_ori, load_image_ori, im_to_numpy
# from .util import isfile, im_to_torch, fliplr, shufflelr_ori, crop_ori, color_normalize, to_torch, transform, draw_labelmap_ori
from PIL import Image

class Animal_Pose_mt_proxy(Animal14KeypointDataset):
    eye = (0, 1)
    hoof = (2, 3, 4, 5)
    knee = (6, 7, 8, 9)
    elbow = (10, 11, 12, 13)
    all = tuple(range(14))

    right_front_leg = (2, 6, 10)
    left_front_leg = (3, 7, 11)
    right_back_leg = (4, 8, 12)
    left_back_leg = (5, 9, 13)
    eyes = (0, 1)

    colored_skeleton = {
        "right_front_leg": (right_front_leg, 'red'),
        "left_front_leg": (left_front_leg, 'orange'),
        "right_back_leg": (right_back_leg, 'green'),
        "left_back_leg": (left_back_leg, 'blue'),
        "eyes": (eyes, 'purple'),
    }
    keypoints_group = {
        "eye": eye,
        "hoof": hoof,
        "knee": knee, 
        "elbow": elbow, 
        "all": all,
    }

    num_keypoints = 14

    def __init__(self, sets=None, is_train=True, is_aug=True, transforms_stu=None, transforms_tea=None, k=1, **kwargs):
        print()
        print("==> animal_pose_mt_proxy")
        self.cat_table = {
            "cow": 1, 
            "sheep": 2, 
            "horse": 3, 
            "cat": 4, 
            "dog": 5
        }
        self.pts_index = np.array([0, 1, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8])
        self.img_folder = kwargs['image_path']  # root image folders
        self.is_train = is_train  # training set or test set
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.animal = ['dog', 'sheep'] if kwargs['animal'] == 'all' else [kwargs['animal']] # train on single or all animal categories
        self.train_on_all_cat = kwargs['train_on_all_cat']  # train on single or mul, decide mean file to load
        self.is_aug = is_aug
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = k
        self.anno_dict = json.load(open(os.path.join(self.img_folder,"animal-pose/keypoints.json")))


        # create train/val split
        if sets is not None:
            self.train_set = sets[0]
            self.val_set = sets[1]
        else:
            self.train_set = []
            self.val_set = []
            self.load_animal()
        # self.mean, self.std = self._compute_mean()
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                sometimes(iaa.Affine(
                    scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                    # scale images to 50-150% of their size, individually per axis
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    # translate by -5 to +5 percent (per axis)
                    rotate=(-30, 30),  # rotate by -30 to +30 degrees
                    shear=(-20, 20),  # shear by -20 to +20 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                sometimes(iaa.AdditiveGaussianNoise(scale=0.5 * 255, per_channel=0.5)),
                sometimes(iaa.GaussianBlur(sigma=(1.0, 5.0))),
                sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),  # improve or worsen the contrast
            ],
            random_order=True
        )


    def load_animal(self):
        # generate train/val data
        for animal in sorted(self.animal):
            train_anno = np.load('./cached_data/real_animal_pose/' + animal + '/train_anno.npy', allow_pickle=True)
            valid_anno = np.load('./cached_data/real_animal_pose/' + animal + '/test_anno.npy', allow_pickle=True)

            self.train_set += train_anno.tolist()
            self.val_set += valid_anno.tolist()
            print('Animal:{}, number of image:{}, train: {}, valid: {}'.format(animal, len(train_anno) + len(valid_anno),
                                                                         len(train_anno), len(valid_anno)))

        print('Total number of image:{}, train: {}, valid {}'.format(len(self.train_set) + len(self.val_set), len(self.train_set),
                                                                      len(self.val_set)))

        """def __getitem__(self, index):

            anno_list = self.train_set if self.is_train else self.val_set
            image_map = self.anno_dict["images"]

            imagename = image_map[str(anno_list[index]["image_id"])]
            image_path = os.path.join(self.img_folder, "animal-pose", "images", imagename)
            img = load_image_ori(image_path)  # CxHxW torch.Size([3, 225, 400])
            # bbox = anno["bbox"]
            # pts = np.array(anno_list[index]["keypoints"]).astype(np.float32)[self.pts_index]
            pts = np.array(anno_list[index]["keypoints"]).astype(np.float32)#[self.pts_index]
            x_min, y_min, x_max, y_max = anno_list[index]["bbox"]

            nparts = pts.shape[0]

            # Generate center and scale for image cropping,
            # adapted from human pose https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/dataset/mpii.lua
            c = torch.Tensor(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))
            s = max(x_max - x_min, y_max - y_min) / 200.0 * 1.25

            # For single-animal pose estimation with a centered/scaled figure
            r = 0
            if self.is_aug and self.is_train:
                # print('augmentation')
                s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
                r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

                # Flip
                if random.random() <= 0.5:
                    img = torch.from_numpy(fliplr(img.numpy())).float()
                    pts = shufflelr_ori(pts, width=img.size(2), dataset='animal_pose')
                    c[0] = img.size(2) - c[0]

                # Color
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

            # Prepare image and groundtruth map
            inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r) # torch.Size([3, 256, 256])
            inp = im_to_numpy(inp * 255).astype(np.uint8)
            inp_stu = Image.fromarray(inp)
            intrinsic_matrix = np.zeros((3,3)) # dummy intrinsic matrix

            inp_stu, data_stu = self.transforms_stu(inp_stu, keypoint2d=pts[:,:2], intrinsic_matrix=intrinsic_matrix)
            pts_stu = data_stu['keypoint2d']
            aug_param_stu = data_stu['aug_param']

            pts_stu = torch.Tensor(pts_stu)
            # image_stu = color_normalize(inp_stu, self.mean, self.std)
            image_stu = inp_stu

            # Generate ground truth
            tpts_stu = pts_stu.clone()
            tpts_ori = torch.Tensor(pts).clone()
            tpts_inpres_stu = pts_stu.clone()
            target_ori = torch.zeros(nparts, self.out_res, self.out_res) # torch.Size([18, 64, 64]) 
            target_stu = torch.zeros(nparts, self.out_res, self.out_res) # torch.Size([18, 64, 64]) 
            target_weight_ori = torch.Tensor(pts[:, 2]).clone().view(nparts, 1) # torch.Size([18, 1])
            target_weight_stu = torch.Tensor(pts[:, 2]).clone().view(nparts, 1) # torch.Size([18, 1])

            for i in range(nparts):
                if tpts_stu[i, 1] > 0:
                    tpts_stu[i, 0:2] = to_torch(transform(tpts_stu[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                    tpts_ori[i, 0:2] = to_torch(transform(tpts_ori[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                    tpts_inpres_stu[i, 0:2] = to_torch(transform(tpts_inpres_stu[i, 0:2] + 1, c, s, [self.inp_res, self.inp_res], rot=r))
                    target_ori[i], vis_ori = draw_labelmap_ori(target_ori[i], tpts_ori[i] - 1, self.sigma, type=self.label_type)
                    target_stu[i], vis_stu = draw_labelmap_ori(target_stu[i], tpts_stu[i] - 1, self.sigma, type=self.label_type)
                    target_weight_stu[i, 0] *= vis_stu
                    target_weight_ori[i, 0] *= vis_ori

            # Meta info
            meta_stu = {'index': index, 'center': c, 'scale': s, 'aug_param_stu': aug_param_stu, 'target_ori': target_ori,
                    'pts': pts_stu, 'tpts': tpts_stu, 'keypoint2d': tpts_inpres_stu, 'target_weight_ori': target_weight_ori }


            return image_stu, target_stu, target_weight_stu, meta_stu"""

    def __getitem__(self, index):

        anno_list = self.train_set if self.is_train else self.val_set
        image_map = self.anno_dict["images"]

        imagename = image_map[str(anno_list[index]["image_id"])]
        image_path = os.path.join(self.img_folder, "animal-pose", "images", imagename)
        # img = load_image_ori(image_path)  # CxHxW torch.Size([3, 225, 400])
        img = np.array(imageio.imread(image_path))[:, :, :3]
        # bbox = anno["bbox"]
        # pts = np.array(anno_list[index]["keypoints"]).astype(np.float32)[self.pts_index]
        pts = np.array(anno_list[index]["keypoints"]).astype(np.float32)
        x_min, y_min, x_max, y_max = anno_list[index]["bbox"]
        # print('b',img.shape, x_min, y_min, x_max, y_max)

        if self.is_aug and self.is_train:
            img_aug = np.expand_dims(img, axis=0)
            # print('b1',img_aug.shape, x_min, y_min, x_max, y_max)

            pts_aug = pts[:, :2].copy()
            pts_aug = np.expand_dims(pts_aug, axis=0)
            img_aug, pts_aug = self.seq(images=img_aug, keypoints=pts_aug)

            img = img_aug.squeeze(0)
            img = im_to_torch(img)
            pts[:, :2] = pts_aug
            pts = torch.Tensor(pts)
            for j in range(pts.size()[0]):
                if pts[j][0] < 0 or pts[j][1] < 0 or pts[j][0] > 640 or pts[j][1] > 480:
                    pts[j] = 0
            # print(pts)

        # print('a',img.shape, x_min, y_min, x_max, y_max)

        nparts = pts.shape[0]

        # Generate center and scale for image cropping,
        # adapted from human pose https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/dataset/mpii.lua
        c = torch.Tensor(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))
        s = max(x_max - x_min, y_max - y_min) / 200.0 * 1.25

        # For single-animal pose estimation with a centered/scaled figure
        r = 0

        # Prepare image and groundtruth map
        inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r) # torch.Size([3, 256, 256])

        # Generate ground truth
        tpts = pts.clone()
        tpts_inpres = pts.clone()
        nparts = tpts.shape[0]
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                tpts_inpres[i, 0:2] = to_torch(transform(tpts_inpres[i, 0:2] + 1, c, s, [self.inp_res, self.inp_res], rot=r))
                target[i], vis = draw_labelmap_ori(target[i], tpts[i] - 1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis
        tpts[:, 2] = target_weight.view(-1)

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'keypoint2d': tpts_inpres}


        return inp, target, target_weight, meta
    
    def __len__(self):
        if self.is_train:
            return len(self.train_set)
        else:
            return len(self.val_set)


def animal_pose_mt_proxy(**kwargs):
    return Animal_Pose_mt_proxy(**kwargs)


animal_pose_mt_proxy.njoints = 14  # ugly but works

