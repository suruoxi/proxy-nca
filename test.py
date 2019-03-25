import torch
import torch.nn as nn
import torch.nn.functional as F
from bottleneck import argpartition
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import os
import torchvision.datasets as datasets
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import resnets as models

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 64)

model.cuda()

#_state_dict = torch.load('checkpoints/best_cub_default_checkpoint.pth.tar')['state_dict']
_state_dict = torch.load('checkpoints/cub_sgd_checkpoint_54.pth.tar')['state_dict']

state_dict = {}

for k,v in _state_dict.items():
    if k.startswith('module.'):
        k = k[7:]
    state_dict[k] = v

model.load_state_dict(state_dict)

model.eval()

tfm = transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

val_dataset = datasets.ImageFolder(
    'cub_pytorch/val',
    tfm)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100,shuffle=False,pin_memory=True)

feats = []
labels = []

with torch.no_grad():
    for x,y in tqdm(val_loader):
        x = x.cuda(None, non_blocking=True)

        feat = model(x).cpu()
        feats.append(feat)
        labels.append(y)
feats = torch.cat(feats, dim=0)
labels = torch.cat(labels, dim=0)

print(feats.shape, labels.shape)

def evaluate(emb, labels):
    emb =emb.detach()
    emb = emb.cuda()
    emb = F.normalize(emb, p=2, dim=-1)
    mm = torch.matmul(emb, emb.t())
    dist = emb.pow(2).sum(dim=1, keepdim=True) + emb.t().pow(2).sum(dim=0, keepdim=True)  - 2*mm
    dist = dist.sqrt()

    dist = dist.cpu()

    res = {}

    for k in [1,2,4,8,16]:
        name = 'Recall@%d' % k
        correct, cnt = 0.0, 0.0
        for i in range(emb.shape[0]):
            dist[i,i] = 1e10
            nns = argpartition(dist[i],k)[:k]
            if any(labels[i] == labels[nn] for nn in nns):
                correct += 1
            cnt += 1
        res[name] = correct/cnt
    print(res)


evaluate(feats, labels)



    

        
