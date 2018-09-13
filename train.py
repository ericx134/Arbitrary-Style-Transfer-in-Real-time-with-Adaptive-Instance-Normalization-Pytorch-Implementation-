from __future__ import print_function
import argparse
import os
import sys
import time
from PIL import Image
from PIL import ImageFile
#from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed 
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import model
from utils import AverageMeter
from sampler import InfiniteSamplerWrapper


def adjust_learning_rate(optimizer, idx):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * idx)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser(description="Distributed Arbitrary Style Transfer (Training script)")
# Basic options
parser.add_argument('--content_dir', default='/mnt/ericx_nfs_mount/data/COCO14/content_train', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='/mnt/ericx_nfs_mount/data/COCO14/style_train', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth',
                    help='Path to pretrained vgg networks as encoder')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--epochs', default=20, type=int, help='Epochs to train')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--num_worksers', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training (default; 1). ')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--world_size', default=1, type=int,
#                     help='number of distributed processes')
# parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist_backend', default='gloo', type=str,
#                     help='distributed backend')


parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated



print("Process {} in action!".format(args.local_rank))

#######################
# Environment Setting #
#######################
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("Using cuda device for training")
    print("CUDA device count: {}".format(
        torch.cuda.device_count()))

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.distributed = args.world_size > 1

if args.distributed:
    assert args.cuda, "Distributed mode requires running with CUDA."
    torch.cuda.set_device(args.local_rank)
    print("Distributed training on device {}...".format(torch.cuda.current_device()))
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)

if args.seed:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': args.num_worksers, 'pin_memory': True} if args.cuda else {}


#######################
# Dataset Preparation #
#######################
assert(os.path.exists(args.content_dir))
assert(os.path.exists(args.style_dir))

train_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomCrop(256),
    transforms.ToTensor()
    ])

style_dataset = datasets.ImageFolder(root=args.style_dir, transform=train_transform)
content_dataset = datasets.ImageFolder(root=args.content_dir, transform=train_transform)

content_sampler = None
# style_sampler = None
if args.distributed:
    content_sampler = torch.utils.data.distributed.DistributedSampler(content_dataset)
    # style_sampler = torch.utils.data.distributed.DistributedSampler(style_dataset)

args.dist_batch_size = int(args.batch_size/torch.distributed.get_world_size()) if args.distributed else args.batch_size
content_loader = torch.utils.data.DataLoader(content_dataset, sampler=content_sampler,
        batch_size=args.dist_batch_size, shuffle=(content_sampler is None), drop_last=True, **kwargs)
style_loader = torch.utils.data.DataLoader(style_dataset, sampler=InfiniteSamplerWrapper(style_dataset),
        batch_size=args.dist_batch_size, **kwargs)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

#writer = SummaryWriter(log_dir=args.log_dir)

decoder = model.decoder
vgg = model.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
net = model.Net(vgg, decoder)
net.train()


if args.cuda:
    net.cuda()
if args.distributed:
    '''
    Wrap model in our version of DistributedDataParallel.
    This must be done AFTER the model is converted to cuda.
    '''
    net = DDP(net)

optimizer = optim.Adam(net.parameters(), lr=args.lr)

batch_time = AverageMeter()
losses = AverageMeter()
data_time = AverageMeter()

for epoch in range(args.epochs):
    if args.distributed:
        content_sampler.set_epoch(epoch)
    end = time.time()
    style_iter = iter(style_loader)
    for batch_idx, (content, _) in enumerate(content_loader, epoch*len(content_loader)):
        adjust_learning_rate(optimizer, batch_idx)
        style, _ = next(style_iter)
        data_time.update(time.time() - end)
        end = time.time()
        if args.cuda:
            content, style = content.cuda(), style.cuda()
        #print(content.size(), style.size())
        content_loss, style_loss = net(content, style)
        loss = args.content_weight*content_loss + args.style_weight*style_loss
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        #print('loss is {}'.format(loss.item()))

        batch_time.update(time.time()- end)
        end = time.time()
        if batch_idx % args.log_interval == 0 and args.local_rank == 0:
            print('Epoch: [{0}][{1}/{2} ({3} in total)]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, batch_idx*len(content), len(content_loader.sampler), len(content_loader.dataset),
                    batch_time=batch_time, data_time=data_time, loss=losses))
            sys.stdout.flush()
        # writer.add_scalar('content_loss', content_loss.item(), batch_idx + 1)
        # writer.add_scalar('loss_style', loss_s.item(), i + 1)




        if (batch_idx + 1) % args.save_model_interval == 0 and args.local_rank == 0:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                '{:s}/decoder_epoch_{:d}_iter_{:d}.pth.tar'.format(args.save_dir, epoch, batch_idx + 1))
#writer.close()