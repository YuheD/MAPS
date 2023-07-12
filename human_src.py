import random
import time
import warnings
import sys
import argparse
import shutil
import os
import shutil
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import lib.models as models
from lib.models.loss import JointsMSELoss, ConsLoss
import lib.datasets as datasets
import lib.transforms.keypoint_detection as T
from lib.transforms import Denormalize
from lib.data import ForeverDataIterator
from lib.meter import AverageMeter, ProgressMeter, AverageMeterDict, AverageMeterList
from lib.keypoint_detection import accuracy
from lib.logger import CompleteLogger
from lib.models import Style_net
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recover_min = torch.tensor([-2.1179, -2.0357, -1.8044]).to(device)
recover_max = torch.tensor([2.2489, 2.4285, 2.64]).to(device)

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log + '_' + args.arch, args.phase)

    logger.write(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    # print("+++++++++++++++++++++++++++")

    # Data loading code
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    src_train_transform = T.Compose([
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
        T.ToTensor(),
        normalize
    ])
   
    base_transform = T.Compose([
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
    ])
    tgt_train_transform_stu = T.Compose([
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
        T.ToTensor(),
        normalize
    ])
    tgt_train_transform_tea = T.Compose([
        T.RandomAffineRotation(args.rotation_tea, args.shear_tea, args.translate_tea, args.scale_tea),
        T.ColorJitter(brightness=args.color_tea, contrast=args.color_tea, saturation=args.color_tea),
        T.GaussianBlur(high=args.blur_tea),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        normalize
    ])
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)
    source_dataset = datasets.__dict__[args.source]

    train_source_dataset = source_dataset(root=args.source_root, transforms=src_train_transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    # print("+++++++++++++++++++++++++++")
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_source_dataset = source_dataset(root=args.source_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_source_loader = DataLoader(val_source_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True)
    # print("+++++++++++++++++++++++++++")

    target_dataset = datasets.__dict__[args.target_train]
    train_target_dataset = target_dataset(root=args.target_root, transforms_base=base_transform,
                                          transforms_stu=tgt_train_transform_stu, transforms_tea=tgt_train_transform_tea, 
                                          k=args.k, image_size=image_size, heatmap_size=heatmap_size)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    target_dataset = datasets.__dict__[args.target]
    val_target_dataset = target_dataset(root=args.target_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_target_loader = DataLoader(val_target_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True)
    # print("+++++++++++++++++++++++++++")
    logger.write("Source train: {}".format(len(train_source_loader)))
    logger.write("Target train: {}".format(len(train_target_loader)))
    logger.write("Source test: {}".format(len(val_source_loader)))
    logger.write("Target test: {}".format(len(val_target_loader)))

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model

    student = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()
    teacher = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()

    
    criterion = JointsMSELoss()
    con_criterion = ConsLoss()

    stu_optimizer = Adam(student.parameters(), lr=args.lr)

    tea_optimizer = OldWeightEMA(teacher, student, alpha=args.teacher_alpha)

    lr_scheduler = MultiStepLR(stu_optimizer, args.lr_step, args.lr_factor)

    student = torch.nn.DataParallel(student).cuda()
    teacher = torch.nn.DataParallel(teacher).cuda()


    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        stu_optimizer.load_state_dict(checkpoint['stu_optimizer'])
        # tea_optimizer.load_state_dict(checkpoint['tea_optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    elif args.pretrain:
        pretrained_dict = torch.load(args.pretrain, map_location='cpu')['student']
        model_dict = student.state_dict()
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        student.load_state_dict(pretrained_dict, strict=False)
        teacher.load_state_dict(pretrained_dict, strict=False)


    # define visualization function
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])

    def visualize(image, keypoint2d, name):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            keypoint2d (tensor): keypoints in shape K x 2
            name: name of the saving image
        """
        train_source_dataset.visualize(tensor_to_image(image),
                                       keypoint2d, logger.get_image_path("{}.jpg".format(name)))

    if args.phase == 'test':
        # evaluate on validation set
        source_val_acc = validate(val_source_loader, teacher, criterion, None, args)
        target_val_acc = validate(val_target_loader, teacher, criterion, visualize, args)

        logger.write("Source: {:4.3f} Target: {:4.3f}".format(source_val_acc['all'], target_val_acc['all']))
        for name, acc in target_val_acc.items():
            logger.write("{}: {:4.3f}".format(name, acc))
        return

    # start training
    best_acc = 0

    for epoch in range(start_epoch, args.pretrain_epoch):
        logger.set_epoch(epoch)
        lr_scheduler.step()

        # train for one epoch
        # if epoch < args.pretrain_epoch:
        pretrain(train_source_iter, student, criterion, stu_optimizer, epoch, visualize if args.debug else None, args)

        # evaluate on validation set
        if epoch < args.pretrain_epoch:
            source_val_acc = validate(val_source_loader, student, criterion, None, args)
            target_val_acc = validate(val_target_loader, student, criterion, visualize if args.debug else None, args)
        else:
            source_val_acc = validate(val_source_loader, teacher, criterion, None, args)
            target_val_acc = validate(val_target_loader, teacher, criterion, visualize if args.debug else None, args)

        # if epoch % 5 == 0:
        #     torch.save(
        #         {
        #             'student': student.state_dict(),
        #             'teacher': teacher.state_dict(),
        #             'stu_optimizer': stu_optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args
        #         }, logger.get_checkpoint_path(f'final_{epoch}_pt')
        #     )
        if not os.path.exists(args.source_out):
            os.makedirs(args.source_out)

        torch.save(
            {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'stu_optimizer': stu_optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }, args.source_out + '/final.pt'
        )
        if target_val_acc['all'] > best_acc:
            best_acc = target_val_acc['all']
            torch.save(
                {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'stu_optimizer': stu_optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
                }, logger.get_checkpoint_path('best_pt')
            )
        logger.write("Epoch: {} Source: {:4.3f} Target: {:4.3f} Target(best): {:4.3f}".format(epoch, source_val_acc['all'], target_val_acc['all'], best_acc))
        for name, acc in target_val_acc.items():
            logger.write("{}: {:4.3f}".format(name, acc))

    logger.close()

def pretrain(train_source_iter, student, criterion, stu_optimizer, epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_all = AverageMeter('Loss (all)', ":.4e")
    losses_s = AverageMeter('Loss (s)', ":.4e")
    acc_s = AverageMeter("Acc (s)", ":3.2f")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_all, losses_s, acc_s],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for i in range(args.iters_per_epoch):
        stu_optimizer.zero_grad()

        x_s, label_s, weight_s, meta_s = next(train_source_iter)
        x_s = x_s.to(device)
        label_s = label_s.to(device)
        weight_s = weight_s.to(device)

        # if style_net is not None and args.s2t_freq > np.random.rand():
        #     with torch.no_grad():
        #         _, _, _, _ , x_ts, _, _ , _= next(train_target_iter)
        #         x_t = x_ts[0].to(device)
        #         _a = np.random.uniform(*args.s2t_alpha)
        #         x_s = style_net(x_s, x_t, _a)[2]
        #         x_s = torch.maximum(torch.minimum(x_s.permute(0,2,3,1), recover_max), recover_min).permute(0,3,1,2)
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast():
            y_s = student(x_s)
            loss_s = criterion(y_s, label_s, weight_s)

        loss_all = loss_s 
        scaler.scale(loss_all).backward()
        scaler.step(stu_optimizer)
        scaler.update()

        _, avg_acc_s, cnt_s, pred_s = accuracy(y_s.detach().cpu().numpy(),
                                               label_s.detach().cpu().numpy())
        acc_s.update(avg_acc_s, cnt_s)
        losses_all.update(loss_all, x_s.size(0))
        losses_s.update(loss_s, x_s.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if visualize is not None:
                visualize(x_s[0], pred_s[0] * args.image_size / args.heatmap_size, "source_{}_pred.jpg".format(i))
                visualize(x_s[0], meta_s['keypoint2d'][0], "source_{}_label.jpg".format(i))

def validate(val_loader, model, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc = AverageMeterList(list(range(val_loader.dataset.num_keypoints)), ":3.2f",  ignore_val=-1)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses], 
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, label, weight, meta) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)
            weight = weight.to(device)

            # compute output
            y = model(x)
            loss = criterion(y, label, weight)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            acc_per_points, avg_acc, cnt, pred = accuracy(y.cpu().numpy(),
                                                          label.cpu().numpy())
            acc.update(acc_per_points, x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.val_print_freq == 0:
                progress.display(i)
                if visualize is not None:
                    visualize(x[0], pred[0] * args.image_size / args.heatmap_size, "val_{}_pred.jpg".format(i))
                    visualize(x[0], meta['keypoint2d'][0], "val_{}_label.jpg".format(i))

    return val_loader.dataset.group_accuracy(acc.average())

     


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Source Only for Keypoint Detection Domain Adaptation')
    # dataset parameters
    parser.add_argument('--source_root', default='human_data/SURREAL', help='root path of the source dataset')
    parser.add_argument('--target_root', default='human_data/LSP', help='root path of the target dataset')
    parser.add_argument('-s', '--source', default='SURREAL', help='source domain(s)')
    parser.add_argument('-t', '--target', default='LSP', help='target domain(s)')
    parser.add_argument('--target-train', default='LSP_mt', help='target domain(s)')
    parser.add_argument('--resize-scale', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--heatmap-size', type=int, default=64,
                        help='output heatmap size')
    parser.add_argument('--sigma', type=int, default=2,
                        help='')
    parser.add_argument('--k', type=int, default=1,
                        help='')

    # augmentation
    parser.add_argument('--rotation_stu', type=int, default=60,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--color_stu', type=float, default=0.25,
                        help='color range of the jitter augmentation')
    parser.add_argument('--blur_stu', type=float, default=0,
                        help='blur range of the jitter augmentation')
    parser.add_argument('--shear_stu', nargs='+', type=float, default=(-30, 30),
                        help='shear range for the RandomResizeCrop augmentation')
    parser.add_argument('--translate_stu', nargs='+', type=float, default=(0.05, 0.05),
                        help='tranlate range for the RandomResizeCrop augmentation')
    parser.add_argument('--scale_stu', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--rotation_tea', type=int, default=60,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--color_tea', type=float, default=0.25,
                        help='color range of the jitter augmentation')
    parser.add_argument('--blur_tea', type=float, default=0,
                        help='blur range of the jitter augmentation')
    parser.add_argument('--shear_tea', nargs='+', type=float, default=(-30, 30),
                        help='shear range for the RandomResizeCrop augmentation')
    parser.add_argument('--translate_tea', nargs='+', type=float, default=(0.05, 0.05),
                        help='tranlate range for the RandomResizeCrop augmentation')
    parser.add_argument('--scale_tea', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--s2t-freq', type=float, default=0.5)
    parser.add_argument('--s2t-alpha', nargs='+', type=float, default=(0, 1))
    parser.add_argument('--t2s-freq', type=float, default=0.5)
    parser.add_argument('--t2s-alpha', nargs='+', type=float, default=(0, 1))

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='pose_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: pose_resnet101)')

    parser.add_argument("--resume", type=str, default=None,
                        help="where restore model parameters from.")
    parser.add_argument("--pretrain", type=str, default=None,
                        help="where restore model parameters from.")
    parser.add_argument("--decoder-name", type=str, default=None,
                        help="where restore style_net model parameters from.")

    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--test-batch', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lambda_c', default=1., type=float)
    parser.add_argument('--teacher_alpha', default=0.999, type=float)
    parser.add_argument('--lr-step', default=[30, 40], type=tuple, help='parameter for lr scheduler')
    parser.add_argument('--lr-factor', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=70, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--val-print-freq', default=2000, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='logs/source/surreal',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', default=1, type=float,
                        help='In the debug mode, save images and predictions')
    parser.add_argument('--pretrain-epoch', type=int, default=50,
                        help='pretrain-epoch')
    parser.add_argument("--source_out", type=str, default='logs/human/source')

    args = parser.parse_args()

    # if args.pretrain_epoch == 50:
    #     args.lr_step = [30, 40]
    if args.pretrain_epoch == 40:
        args.lr_step = [45, 60]
    main(args)

