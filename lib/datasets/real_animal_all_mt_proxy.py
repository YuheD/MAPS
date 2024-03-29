from __future__ import print_function, absolute_import

import random
import torch.utils.data as data
# from pose.utils.osutils import *
# from pose.utils.transforms import *

from scipy.io import loadmat
import argparse
from .keypoint_dataset import Animal18KeypointDataset
from .util import isfile
import os
import numpy as np
import torch
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from .util import isfile, im_to_torch, fliplr, shufflelr_ori, crop_ori, color_normalize, to_torch, transform, draw_labelmap_ori, load_image_ori, im_to_numpy, crop_ori_256
from PIL import Image

class Real_Animal_All_mt_proxy(Animal18KeypointDataset):
    eye = (0, 1)
    chin = (2,)
    hoof = (3, 4, 5, 6)
    hip = (7,)
    knee = (8, 9, 10, 11)
    shoulder = (12, 13)
    elbow = (14, 15, 16, 17)
    all = tuple(range(18))

    right_front_leg = (3, 8, 14)
    left_front_leg = (4, 9, 15)
    right_back_leg = (5, 10, 16)
    left_back_leg = (6, 11, 17)
    right_torso = (13, 7)
    right_face = (1, 2)
    left_torso = (12, 7)
    left_face = (0, 2)

    num_keypoints = 18

    colored_skeleton = {
        "right_front_leg": (right_front_leg, 'red'),
        "left_front_leg": (left_front_leg, 'orange'),
        "right_back_leg": (right_back_leg, 'green'),
        "left_back_leg": (left_back_leg, 'red'),
        "right_torso": (right_torso, 'blue'),
        "right_face": (right_face, 'blue'),
        "left_torso": (left_torso, 'purple'),
        "left_face": (left_face, 'purple'),
    }
    keypoints_group = {
        "eye": eye,
        "chin": chin,
        "hoof": hoof,
        "hip": hip,
        "knee": knee, 
        "shoulder": shoulder,
        "elbow": elbow, 
        "all": all,
    }

    def __init__(self, sets=None, is_train=True, is_tune=False, transforms_stu=None, transforms_tea=None, k=1, **kwargs):
        # print()
        print("==> real_animal_all_mt_proxy")
        self.img_folder = kwargs['image_path']  # root image folders
        self.is_train = is_train  # training set or test set
        self.is_tune = is_tune
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.label_type = kwargs['label_type']
        self.animal = ['horse', 'tiger'] if kwargs['animal'] == 'all' else [kwargs['animal']] # train on single or all animal categories
        self.train_on_all_cat = kwargs['train_on_all_cat']  # train on single or mul, decide mean file to load

        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = k

        # create train/val split
        self.train_img_set = []
        self.valid_img_set = []
        self.train_pts_set = []
        self.valid_pts_set = []
        if self.is_tune:
            self.tune_img_set = []
            self.tune_pts_set = []
        if sets is not None:
            self.train_img_set = sets[0]
            self.valid_img_set = sets[1]
            self.train_pts_set = sets[2]
            self.valid_pts_set = sets[3]
            if self.is_tune:
                self.tune_img_set = sets[4]
                self.tune_pts_set = sets[5]
            # self.tgt_set = sets[2]
        else:
            self.load_animal()

        # self.mean, self.std = self._compute_mean()

        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        # self.seq_stu = iaa.Sequential(
        #     [
        #         # sometimes(iaa.AdditiveGaussianNoise(scale=0.5 * 20, per_channel=0.5)),
        #         # sometimes(iaa.GaussianBlur(sigma=(1.0, 5.0))),
        #         sometimes(iaa.ContrastNormalization((1, 1), per_channel=0.5)),  # improve or worsen the contrast
        #     ],
        #     random_order=True
        # )

        # self.seq_tea = iaa.Sequential(
        #     [
        #         # sometimes(iaa.AdditiveGaussianNoise(scale=0.5 * 10, per_channel=0.5)),
        #         # sometimes(iaa.GaussianBlur(sigma=(1.0, 5.0))),
        #         sometimes(iaa.ContrastNormalization((1, 1), per_channel=0.5)),  # improve or worsen the contrast
        #     ],
        #     random_order=True
        # )
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
            img_list = []  # img_list contains all image paths
            anno_list = []  # anno_list contains all anno lists
            range_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0/ranges', animal, 'ranges.mat')
            landmark_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0/landmarks', animal)
            range_file = loadmat(range_path)
            frame_num = 0

            train_idxs = np.load('./cached_data/real_animal/' + animal + '/train_idxs_by_video.npy')
            valid_idxs = np.load('./cached_data/real_animal/' + animal + '/valid_idxs_by_video.npy')
            for video in range_file['ranges']:
                # range_file['ranges'] is a numpy array [Nx3]: shot_id, start_frame, end_frame
                shot_id = video[0]
                landmark_path_video = os.path.join(landmark_path, str(shot_id) + '.mat')

                if not os.path.isfile(landmark_path_video):
                    continue
                landmark_file = loadmat(landmark_path_video)

                for frame in range(video[1], video[2] + 1):  # ??? video[2]+1
                    frame_id = frame - video[1]
                    img_name = animal + '/' + '0' * (8 - len(str(frame))) + str(frame) + '.jpg'
                    img_list.append([img_name, shot_id, frame_id])

                    coord = landmark_file['landmarks'][frame_id][0][0][0][0]
                    vis = landmark_file['landmarks'][frame_id][0][0][0][1]
                    landmark = np.hstack((coord, vis))
                    landmark_18 = landmark[:18, :]
                    if animal == 'horse':
                        anno_list.append(landmark_18)
                    elif animal == 'tiger':
                        landmark_18 = landmark_18[
                            np.array([1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 13, 14, 9, 10, 11, 12]) - 1]
                        anno_list.append(landmark_18)
                    frame_num += 1

            for idx in range(train_idxs.shape[0]):
                train_idx = train_idxs[idx]
                if self.is_tune and idx % 5 ==0:
                    self.tune_img_set.append(img_list[train_idx])
                    self.tune_pts_set.append(anno_list[train_idx])
                else:
                    self.train_img_set.append(img_list[train_idx])
                    self.train_pts_set.append(anno_list[train_idx])
            for idx in range(valid_idxs.shape[0]):
                valid_idx = valid_idxs[idx]
                self.valid_img_set.append(img_list[valid_idx])
                self.valid_pts_set.append(anno_list[valid_idx])
            
        #     print('Animal:{}, number of frames:{}, train: {}, valid: {}'.format(animal, frame_num,
        #                                                                 train_idxs.shape[0], valid_idxs.shape[0]))
        # if self.is_tune:
        #     print('Total number of frames:{}, train: {}, tune: {}, valid {}'.format(len(img_list), len(self.train_img_set), len(self.tune_img_set),
        #                                                                 len(self.valid_img_set)))
        # else:
        #     print('Total number of frames:{}, train: {}, valid {}'.format(len(img_list), len(self.train_img_set),
        #                                                                 len(self.valid_img_set)))

    def __getitem__(self, index):

        if self.is_train:
            img_list = self.train_img_set
            anno_list = self.train_pts_set
        else:
            img_list = self.tune_img_set if self.is_tune else self.valid_img_set
            anno_list = self.tune_pts_set if self.is_tune else self.valid_pts_set
        try:
            a = img_list[index][0]
        except IndexError:
            print(index)

        img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0',  a)
        pts = anno_list[index].astype(np.float32) # (18, 3)



        pts_aug = pts[:, :2].copy()
        img = np.array(imageio.imread(img_path))[:, :, :3]
        # print('b',img.shape)
        img_aug = np.expand_dims(img, axis=0)
        pts_aug = np.expand_dims(pts_aug, axis=0)
        # if self.is_train and self.is_aug:
        # print(pts)
        img_aug, pts_aug = self.seq(images=img_aug, keypoints=pts_aug)
        img = img_aug.squeeze(0)
        img = im_to_torch(img)
        pts[:, :2] = pts_aug

        # print('a',img.size())
        # ezit(0)

        x_vis = pts[:, 0][pts[:, 0] > 0]
        y_vis = pts[:, 1][pts[:, 1] > 0]
        nparts = pts.shape[0]
        # print(pts.shape)
        # print(img.size())
        # print(x_vis)
        # print(y_vis)

        try:
            # generate bounding box using keypoints
            height, width = img.size()[1], img.size()[2]
            y_min = float(max(np.min(y_vis) - 15, 0.0))
            y_max = float(min(np.max(y_vis) + 15, height))
            x_min = float(max(np.min(x_vis) - 15, 0.0))
            x_max = float(min(np.max(x_vis) + 15, width))
        except ValueError:
            print(img_path, index)
        # Generate center and scale for image cropping,
        # adapted from human pose https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/dataset/mpii.lua
        # c = torch.Tensor((256 / 2.0, 256 / 2.0))
        c = torch.Tensor(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))
        s = max(x_max - x_min, y_max - y_min) / 200.0 * 1.25

        # For single-animal pose estimation with a centered/scaled figure
        r = 0

        # Prepare image and groundtruth map

        inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r) # torch.Size([3, 256, 256])
        # inp = color_normalize(inp, self.mean, self.std)
        # Generate ground truth
        pts = torch.Tensor(pts)
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

        meta = {'index': index, 'center': c, 'scale': s,
            'pts': pts, 'tpts': tpts, 'keypoint2d': tpts_inpres}
        return inp, target, target_weight, meta
    

    """def __getitem__(self, index):

        if self.is_train:
            img_list = self.train_img_set
            anno_list = self.train_pts_set
        else:
            img_list = self.tune_img_set if self.is_tune else self.valid_img_set
            anno_list = self.tune_pts_set if self.is_tune else self.valid_pts_set
        try:
            a = img_list[index][0]
        except IndexError:
            print(index)

        img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', a)
        img = load_image_ori(img_path)  # CxHxW torch.Size([3, 225, 400])
        pts = anno_list[index].astype(np.float32) # (18, 3)
        x_vis = pts[:, 0][pts[:, 0] > 0]
        y_vis = pts[:, 1][pts[:, 1] > 0]
        nparts = pts.shape[0]
        # print(x_vis)
        # print(y_vis)
        # exit(0)

        try:
            # generate bounding box using keypoints
            height, width = img.size()[1], img.size()[2]
            y_min = float(max(np.min(y_vis) - 15, 0.0))
            y_max = float(min(np.max(y_vis) + 15, height))
            x_min = float(max(np.min(x_vis) - 15, 0.0))
            x_max = float(min(np.max(x_vis) + 15, width))
        except ValueError:
            print(img_path, index)
        # Generate center and scale for image cropping,
        # adapted from human pose https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/dataset/mpii.lua
        c = torch.Tensor(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))
        s = max(x_max - x_min, y_max - y_min) / 200.0 * 1.25

        # For single-animal pose estimation with a centered/scaled figure
        r = 0
        # Prepare image and groundtruth map
        inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r) # torch.Size([3, 256, 256])

        inp = im_to_numpy(inp * 255).astype(np.uint8)
        # inp_stu = self.seq_stu(images=inp)
        inp_stu = Image.fromarray(inp)

        intrinsic_matrix = np.zeros((3,3)) # dummy intrinsic matrix
        # print(pts[:,:2])
        # exit(0)
        inp_stu, data_stu = self.transforms_stu(inp_stu, keypoint2d=pts[:,:2], intrinsic_matrix=intrinsic_matrix)
        pts_stu = data_stu['keypoint2d']
        aug_param_stu = data_stu['aug_param']
        # print(pts_stu)
        # print(kp)
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

        return image_stu, target_stu, target_weight_stu, meta_stu#, images_tea, targets_tea, target_weights_tea, metas_tea"""

    def __len__(self):
        if self.is_train:
            return len(self.train_img_set)
        else:
            return len(self.tune_img_set) if self.is_tune else len(self.valid_img_set)


def real_animal_all_mt_proxy(**kwargs):
    return Real_Animal_All_mt_proxy(**kwargs)


real_animal_all_mt_proxy.njoints = 18  # ugly but works

