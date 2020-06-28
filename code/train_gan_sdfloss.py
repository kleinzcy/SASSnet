import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_sdf import VNet
from networks.discriminator import FC3DDiscriminator

from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT_001', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4, help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5, help='balance factor to control supervised and consistency loss')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.01, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True #
    cudnn.deterministic = False #
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes-1, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    D = FC3DDiscriminator(num_classes=num_classes - 1)
    D = D.cuda()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train', # train/val split
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

    model.train()

    Dopt = optim.Adam(D.parameters(), lr=args.D_lr, betas=(0.9,0.99))
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # Generate Discriminator target based on sampler
            Dtarget = torch.tensor([1, 1, 0, 0]).cuda()
            model.train()
            D.eval()

            outputs_tanh, outputs = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)

            ## calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:].cpu().numpy(), outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)
            loss_seg = ce_loss(outputs[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            supervised_loss = loss_seg_dice + args.beta * loss_sdf

            Doutputs = D(outputs_tanh[labeled_bs:], volume_batch[labeled_bs:])
            # G want D to misclassify unlabel data to label data.
            loss_adv = F.cross_entropy(Doutputs, (Dtarget[:labeled_bs]).long())

            loss = supervised_loss + consistency_weight*loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(torch.argmax(outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])

            # Train D
            model.eval()
            D.train()
            with torch.no_grad():
                outputs_tanh, outputs = model(volume_batch)

            Doutputs = D(outputs_tanh, volume_batch)
            # D want to classify unlabel data and label data rightly.
            D_loss = F.cross_entropy(Doutputs, Dtarget.long())

            # Dtp and Dfn is unreliable because of the num of samples is small(4)
            Dacc = torch.mean((torch.argmax(Doutputs, dim=1).float()==Dtarget.float()).float())
            Dtp = torch.mean((torch.argmax(Doutputs, dim=1).float()==Dtarget.float()).float())
            Dfn = torch.mean((torch.argmax(Doutputs, dim=1).float()==Dtarget.float()).float())
            Dopt.zero_grad()
            D_loss.backward()
            Dopt.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/loss_adv', consistency_weight*loss_adv, iter_num)
            writer.add_scalar('GAN/loss_adv', loss_adv, iter_num)
            writer.add_scalar('GAN/D_loss', D_loss, iter_num)
            writer.add_scalar('GAN/Dtp', Dtp, iter_num)
            writer.add_scalar('GAN/Dfn', Dfn, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_weight: %f, loss_haus: %f, loss_seg: %f, loss_dice: %f' %
                (iter_num, loss.item(), consistency_weight, loss_sdf.item(),
                 loss_seg.item(), loss_seg_dice.item()))

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
