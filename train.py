import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from torch.utils.tensorboard import SummaryWriter
from models.MVSNet import MVSNet, mvsnet_loss
from utils import *
import datetime

cudnn.benchmark = True
torch.cuda.empty_cache()

###########################################################
######################## Arguments ########################
###########################################################

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--trainpath', type=str, default="../../Datasets/dtu_training/mvs_training/dtu/", help='train datapath')
parser.add_argument('--testpath', default=None, help='test datapath')
parser.add_argument('--trainlist', type=str, default="lists/dtu/train.txt", help='train list')
parser.add_argument('--testlist', type=str, default="lists/dtu/test.txt", help='test list')

parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=128, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/d52623', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_false', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

args = parser.parse_args()

mode = args.mode
loadckpt = args.loadckpt
resume = False #args.resume
batch_size = args.batch_size
interval_scale = args.interval_scale
lr = args.lr
lr_epochs = args.lrepochs
save_freq = args.save_freq
summary_freq = args.summary_freq
trainpath = args.trainpath
testpath = None
trainlist = args.trainlist
testlist = args.testlist
numdepth = args.numdepth
logdir = args.logdir
seed = args.seed
weight_decay = args.wd
epochs = args.epochs
###########################################################
######################## Arguments ########################
###########################################################

# parse arguments and check
if resume:
    assert mode == "train"
    assert loadckpt is None
if testpath is None:
    testpath = trainpath

def train(trainDataset, model, optimizer, criterion, writer, start_epoch):
    trainImgLoader = DataLoader(trainDataset, batch_size, shuffle=True, num_workers=1, drop_last=True)

    milestones = [int(epoch_idx) for epoch_idx in lr_epochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(lr_epochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

    model.train()

    for epoch in range(start_epoch, epochs):
        print('Epoch {}:'.format(epoch))
        epoch_start = time.time()
        avg_loss = 0.0

        for batch, sample in enumerate(trainImgLoader):
            start_time = time.time()

            sample = tocuda(sample)
            depth_gt = sample["depth"]
            mask = sample["mask"]

            optimizer.zero_grad()

            outputs = model(sample["imgs"], sample["proj_matrices"], sample["depth_values"])
            depth_est = outputs["depth"]

            loss = criterion(depth_est, depth_gt, mask)
            loss.backward()
            optimizer.step()

            batch_loss = tensor2float(loss)
            avg_loss += batch_loss

            if batch % summary_freq == 0:
                print(
                    'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch+1, epochs, batch,
                                                                                    len(trainImgLoader), batch_loss,
                                                                                    time.time() - start_time))
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(trainImgLoader) + batch)

        avg_loss /= len(trainImgLoader)
        print(
            'Epoch Statistics: Epoch {}/{}, average train loss = {:.3f}, time = {:.3f}'.format(epoch+1,
                                                                            epochs, avg_loss,
                                                                            time.time() - epoch_start))
        lr_scheduler.step()
        save_checkpoint(model, optimizer, epoch, logdir)

@make_nograd_func
def test(testDataset, model, criterion, writer):
    testImgLoader = DataLoader(testDataset, batch_size, shuffle=False, num_workers=1, drop_last=False)
    avg_test_scalars = DictAverageMeter()

    model.eval()

    with torch.no_grad():
        for batch, sample in enumerate(testImgLoader):
            start_time = time.time()

            sample = tocuda(sample)
            depth_gt = sample["depth"]
            mask = sample["mask"]

            
            outputs = model(sample["imgs"], sample["proj_matrices"], sample["depth_values"])
            depth_est = outputs["depth"]

            loss = criterion(depth_est, depth_gt, mask)

            scalar_outputs = {"loss": loss}
            image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                            "ref_img": sample["imgs"][:, 0],
                            "mask": sample["mask"]}
            image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

            scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
            scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
            scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
            scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch, len(testImgLoader), loss,
                                                                        time.time() - start_time))
            writer.add_scalar('Test/Loss', loss, batch)
            if batch % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch, len(testImgLoader), avg_test_scalars.mean()))
    
    print("final", avg_test_scalars)



def main():
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    print("Launch Time: ", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # dataset, dataloader
    trainDataset = MVSDataset(trainpath, trainlist, "train", 5, numdepth, interval_scale)
    testDataset = MVSDataset(testpath, testlist, "test", 3, numdepth, interval_scale)

    model = MVSNet(refine=False).to('cuda')

    criterion = mvsnet_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    logger = SummaryWriter(logdir)

    start_epoch = 0

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if mode == 'train':
        if resume == True:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, logdir)

        train(trainDataset, model, optimizer, criterion, logger, start_epoch)

    if mode == 'test':
        if loadckpt == None:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, logdir)
        else:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, logdir, loadckpt)

        test(testDataset, model, criterion, logger)

if __name__ == '__main__':
    main()