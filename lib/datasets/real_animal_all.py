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
from .util import isfile, fliplr, shufflelr_ori, crop_ori, color_normalize, to_torch, transform, draw_labelmap_ori, load_image_ori

class Real_Animal_All(Animal18KeypointDataset):
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

    def __init__(self, is_train=True, is_tune=False, no_norm=False, sets=None, cs=None,  **kwargs):
        print()
        print("==> real_animal_all")
        self.img_folder = kwargs['image_path']  # root image folders
        self.is_train = is_train  # training set or test set
        self.is_tune = is_tune  # training set or test set
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.animal = ['horse', 'tiger'] if kwargs['animal'] == 'all' else [kwargs['animal']] # train on single or all animal categories
        self.train_on_all_cat = kwargs['train_on_all_cat']  # train on single or mul, decide mean file to load
        self.no_norm = no_norm

        # create train/val split
        self.train_img_set = []
        self.valid_img_set = []
        self.train_pts_set = []
        self.valid_pts_set = []
        if self.is_tune:
            self.tune_img_set = []
            self.tune_pts_set = []

        if sets is not None:
            print(sets is None)
            self.train_img_set = sets[0]
            self.valid_img_set = sets[1]
            self.train_pts_set = sets[2]
            self.valid_pts_set = sets[3]
            if self.is_tune:
                self.tune_img_set = sets[4]
                self.tune_pts_set = sets[5]
        else:
            self.load_animal()

        if cs is not None:
            self.cbank=cs[0]
            self.sbank=cs[1]
        else:
            self.cbank=None
            self.sbank=None

        # self.mean, self.std = self._compute_mean()

    def load_animal(self):
        # raise RuntimeError
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
            
            print('Animal:{}, number of frames:{}, train: {}, valid: {}'.format(animal, frame_num,
                                                                        train_idxs.shape[0], valid_idxs.shape[0]))
        if self.is_tune:
            print('Total number of frames:{}, train: {}, tune: {}, valid {}'.format(len(img_list), len(self.train_img_set), len(self.tune_img_set),
                                                                        len(self.valid_img_set)))
        else:
            print('Total number of frames:{}, train: {}, valid {}'.format(len(img_list), len(self.train_img_set),
                                                                        len(self.valid_img_set)))

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
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

        if self.cbank is None:
            x_vis = pts[:, 0][pts[:, 0] > 0]
            y_vis = pts[:, 1][pts[:, 1] > 0]
            # print(img.size(), x_vis, y_vis)
            # exit(0)

            try:
                # generate bounding box using keypoints
                height, width = img.size()[1], img.size()[2] # 225 400
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
            # print(x_max, x_min, y_max, y_min, c, s)
        else:
            c = self.cbank[index]
            s = self.sbank[index]

        # For single-animal pose estimation with a centered/scaled figure
        nparts = pts.shape[0]
        pts = torch.Tensor(pts)
        r = 0
        # if self.is_aug and self.is_train:
        #     # print('augmentation')
        #     s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
        #     r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

        #     # Flip
        #     if random.random() <= 0.5:
        #         img = torch.from_numpy(fliplr(img.numpy())).float()
        #         pts = shufflelr_ori(pts, width=img.size(2), dataset='real_animal')
        #         c[0] = img.size(2) - c[0]

        #     # Color
        #     img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp_ori = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r) # torch.Size([3, 256, 256])

        # inp = color_normalize(inp_ori, self.mean, self.std)
        inp = inp_ori
        # print(inp.size())
        # exit(0)
        # raise RuntimeError()
        # Generate ground truth
        tpts = pts.clone()
        tpts_inpres = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res) # torch.Size([18, 64, 64]) 
        target_weight = tpts[:, 2].clone().view(nparts, 1) # torch.Size([18, 1])

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                # print()
                # print( tpts_inpres[i, 0:2])
                tpts_inpres[i, 0:2] = to_torch(transform(tpts_inpres[i, 0:2] + 1, c, s, [self.inp_res, self.inp_res], rot=r))
                # print( tpts_inpres[i, 0:2])
                tmp = to_torch(transform(tpts_inpres[i, 0:2] + 1, c, s, [self.inp_res, self.inp_res], invert=1, rot=r))
                # print( tmp-1)
                target[i], vis = draw_labelmap_ori(target[i], tpts[i] - 1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis
                # print()

        # exit(0)

        # Meta info
        # bbox = [ x_min, y_min, x_max, y_max ]
        meta = {'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'keypoint2d': tpts_inpres}
        if self.no_norm:
            return inp, inp_ori, target, target_weight, meta
        return inp, target, target_weight, meta # torch.Size([3, 256, 256]) torch.Size([18, 64, 64])

    def __len__(self):
        if self.is_train:
            return len(self.train_img_set)
        else:
            return len(self.tune_img_set) if self.is_tune else len(self.valid_img_set)


def real_animal_all(**kwargs):
    return Real_Animal_All(**kwargs)


real_animal_all.njoints = 18  # ugly but works

