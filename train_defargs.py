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

mode = "train"
loadckpt = None
resume = True
batch_size = 4
interval_scale = 1.06
lr = 0.001
lr_epochs = "10,12,14:2"
save_freq = 1
summary_freq = 20
trainpath = "../../Datasets/dtu_training/mvs_training/dtu/"
testpath = None
trainlist = "lists/dtu/train.txt"
testlist = "lists/dtu/test.txt"
numdepth = 64
logdir = "./checkpoints/d52623"
seed = 1
weight_decay = 0.0
epochs = 20

# parse arguments and check
if resume:
    assert mode == "train"
    assert loadckpt is None
if testpath is None:
    testpath = trainpath

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# create logger for mode "train" and "testall"
if mode == "train":
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(logdir)

# dataset, dataloader
trainDataset = MVSDataset(trainpath, trainlist, "train", 3, numdepth, interval_scale)
testDataset = MVSDataset(testpath, testlist, "test", 3, numdepth, interval_scale)
trainImgLoader = DataLoader(trainDataset, batch_size, shuffle=True, num_workers=1, drop_last=True)
testImgLoader = DataLoader(testDataset, batch_size, shuffle=False, num_workers=1, drop_last=False)

# model, optimizer
model = MVSNet(refine=False)
model.to(torch.device('cuda:0'))
criterion = mvsnet_loss
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

# load parameters
start_epoch = 0
if (mode == "train" and resume) or (mode == "test" and not loadckpt):
    saved_models = [fn for fn in os.listdir(logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # use the latest checkpoint file
    loadckpt = os.path.join(logdir, saved_models[-1])

    print("loading checkpoint ", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif loadckpt:
    # load checkpoint file specified by args.loadckpt
    # no need to load optimizer and epoch because we are invoking in test mode
    print("loading checkpoint {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])

print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in lr_epochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(lr_epochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(trainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(trainImgLoader):
            start_time = time.time()
            global_step = len(trainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, epochs, batch_idx,
                                                                                     len(trainImgLoader), loss,
                                                                                     time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % save_freq == 0:
            save_checkpoint(model, optimizer, epoch_idx, logdir)

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(testImgLoader):
            start_time = time.time()
            global_step = len(trainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, epochs, batch_idx,
                                                                                     len(testImgLoader), loss,
                                                                                     time.time() - start_time))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())

def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(testImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(testImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(testImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)

def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss = criterion(depth_est, depth_gt, mask)
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss = criterion(depth_est, depth_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

if __name__ == '__main__':
    if mode == "train":
        train()
    elif mode == "test":
        test()