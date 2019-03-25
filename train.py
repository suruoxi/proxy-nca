import argparse
import logging
import time
import os
import random
import warnings

import numpy as np
from bottleneck import argpartition

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnets as models
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import MultiStepLR

from proxynca import ProxyNCAUnstable, ProxyNCA

logging.basicConfig(level=logging.INFO)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='train a model for image classification.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data-path', type=str,required=True,
                    help='path of data, which contains train,val subdirectory')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='dimensionality of image embedding. default is 128.')
parser.add_argument('--classes', type=int, required=True,
                    help='number of classes in dataset')
parser.add_argument('--batch-size', type=int, default=128,
                    help='total batch_size on all gpus.')
parser.add_argument('--gpus', type=str, default='0',
                    help='list of gpus to use, e.g. 0 or 0,2,5.')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs. default is 20.')
parser.add_argument('--lr', type=float, default=0.04,
                    help='learning rate. default is 0.0001.')
parser.add_argument('--smoothing-const', type=float, default=0.1,
                    help='default is 0.1.')
parser.add_argument('--factor', type=float, default=0.94,
                    help='learning rate schedule factor. default is 0.94.')
parser.add_argument('--base-eps', type=float, default=1.0,
                    help='eps for base net in Adam')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed to use')
parser.add_argument('--model', type=str, default='resnet50',choices=model_names,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--save-prefix', type=str,required=True,
                    help='prefix of saved checkpoint.')
parser.add_argument('--optimizer', type=str,default='adam',
                    help='which optimizer to use. default is adam.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model.')
parser.add_argument('--exclude-positive', action='store_true',
                    help='exclude-positive')
parser.add_argument('--print-freq', type=int, default=20,
                    help='number of batches to wait before logging.')
args = parser.parse_args()

logging.info(args)

if not os.path.exists('checkpoints/'):
    os.mkdir('checkpoints')

# seed
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('''You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.''')


# gpus setting
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# construct model

if not args.use_pretrained:
    model = models.__dict__[args.model](num_classes=args.embed_dim)
else:
    model = models.__dict__[args.model](pretrained=True)
    try:
        model.fc = nn.Linear(model.fc.in_features, args.embed_dim)
    except NameError as e:
        print("Error: current works only with model having fc layer as the last layer, try modify the code")
        exit(-1)


#print(model.state_dict().keys())
model.cuda()

if args.exclude_positive:
    criterion = ProxyNCAUnstable(args.classes, args.embed_dim, smoothing_const=args.smoothing_const, exclude_positive = True)
else:
    criterion = ProxyNCA(args.classes, args.embed_dim, args.smoothing_const)

#optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, 
#
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(
            [ { # basenet 
                'params': list(set(model.parameters()).difference(set(model.fc.parameters()))),
                'eps': args.base_eps,
                'lr': args.lr
              },
              {# embedding layer
                'params': model.fc.parameters(),
                'lr': args.lr
              },
              {# proxies
                 'params': criterion.parameters(),
                 'lr:': args.lr
              }
            ], lr=args.lr, weight_decay=0.0)
    scheduler = ExponentialLR(optimizer=optimizer,gamma = args.factor)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(
            [ { # basenet 
                'params': list(set(model.parameters()).difference(set(model.fc.parameters()))),
                'lr': args.lr
              },
              {# embedding layer
                'params': model.fc.parameters(),
                'lr': args.lr*2
              },
              {# proxies
                 'params': criterion.parameters(),
                 'lr:': args.lr*2
              }
            ], lr=args.lr, weight_decay=0.0004,momentum=0.9)
    scheduler = MultiStepLR(optimizer,[50,100,150,200],args.factor)


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        state_dict = {}
        for k,v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
        if args.optimizer == 'adam':
            print(">>>>> currently has bug to load Adam state dict <<<<")
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        criterion.proxies = checkpoint['proxies']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


if len(args.gpus.split(',')) > 1:
    model = torch.nn.DataParallel(model)



# dataset 
traindir = os.path.join(args.data_path, 'train')
valdir = os.path.join(args.data_path, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    )

#batch_sampler = BalancedBatchSampler(train_dataset, args.batch_size, args.batch_k, length=args.batch_num)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    num_workers=args.workers,
    shuffle=True,
    batch_size=args.batch_size,
    pin_memory=True
    )


def train(train_loader, model, criterion, optimizer,  epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x,y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        y = y.cuda(None, non_blocking=True)
        x = x.cuda(None, non_blocking=True)

        # compute output
        feat = model(x)
        loss = criterion(feat,y)
        losses.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints/')
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate
        # 

        state = {
            'epoch': epoch+1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'proxies': criterion.proxies
            }
        torch.save(state, 'checkpoints/%s_checkpoint_%d.pth.tar'%(args.save_prefix,epoch+1))
