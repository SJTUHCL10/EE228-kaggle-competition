# -*- coding: utf-8 -*-

data_dir='./sjtu-ee228-2020/'
model_dir='./model/model1.pth'

#%%
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
#import torch.utils.checkpoint as cp
import csv
from sklearn import metrics
import math
import random

train_num = 465
train_num1 = 372
train_num2 = 93
test_num = 117
batch_size = 8
epoch_num = 50
mask_weight = 0.1
alpha = 0.7
displacement = True

class m3dvDataset(data.Dataset):
    def __init__(self,data_dir,label_dir,train,displacement=False,vox_transform=transforms.ToTensor(),seg_transform=transforms.ToTensor()):
        self.data_dir=data_dir
        self.train=train
        self.displacement=displacement
        tmp=np.loadtxt(label_dir,dtype=str,delimiter=",",skiprows=1)
        self.label={}
        for i in range(tmp.shape[0]):
            self.label[tmp[i,0]+'.npz']=int(tmp[i,1])
        self.vox_transform=vox_transform
        self.seg_transform=seg_transform
        self.npz=os.listdir(data_dir)
        self.npz.sort()
    def __getitem__(self,index):
        #global cnt
        if self.train:
            if self.displacement:
                dx=np.random.randint(9)-4
                dy=np.random.randint(9)-4
                dz=np.random.randint(9)-4
                #cnt+=1
                voxel=np.load(self.data_dir+self.npz[index])['voxel'][34+dx:66+dx,34+dy:66+dy,34+dz:66+dz]
                seg=np.load(self.data_dir+self.npz[index])['seg'][34+dx:66+dx,34+dy:66+dy,34+dz:66+dz]
            else:
                voxel=np.load(self.data_dir+self.npz[index])['voxel'][34:66,34:66,34:66]
                seg=np.load(self.data_dir+self.npz[index])['seg'][34:66,34:66,34:66]
            voxel=self.vox_transform(voxel)
            voxel=torch.unsqueeze(voxel, 0)
            seg=self.seg_transform(seg)
            seg=seg.long()
            label=self.label[self.npz[index]]
            return voxel,seg,label
        else:
            index=index+372
            voxel=np.load(self.data_dir+self.npz[index])['voxel'][34:66,34:66,34:66]
            voxel=self.vox_transform(voxel)
            voxel=torch.unsqueeze(voxel, 0)
            seg=np.load(self.data_dir+self.npz[index])['seg'][34:66,34:66,34:66]
            seg=self.seg_transform(seg)
            seg=seg.long()
            label=self.label[self.npz[index]]
            return voxel,seg,label
    def __len__(self):
        if self.train==1:
            return 372
        elif self.train==0:
            return 93
        else:
            return 465
    

def get_mean_and_std():
    dataset=m3dvDataset(data_dir+'train_val/train_val/',data_dir+'train_val.csv',train=1)
    trains = [d[0].data.numpy() for d in dataset]
    return np.mean(trains),np.std(trains)

mean,std=get_mean_and_std()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vox_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
])

seg_transform=transforms.Compose([
    transforms.ToTensor(),
])

class _denselayer(nn.Module):
    def __init__(self,channel_size,bn_size=4,growth_rate=16):
        super(_denselayer,self).__init__()
        self.BN1=nn.BatchNorm3d(channel_size)
        self.ReLU1=nn.ReLU(inplace=True)
        self.Conv1=nn.Conv3d(channel_size,bn_size*growth_rate,kernel_size=1,stride=1,padding=0,bias=True)
        self.BN2=nn.BatchNorm3d(bn_size*growth_rate)
        self.ReLU2=nn.ReLU(inplace=True)
        self.Conv2=nn.Conv3d(bn_size*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=True)
        
    def forward(self,x):
        x=torch.cat(x, 1)
        x=self.BN1(x)
        x=self.ReLU1(x)
        x=self.Conv1(x)
        x=self.BN2(x)
        x=self.ReLU2(x)
        x=self.Conv2(x)
        return x

class _denseblock(nn.ModuleDict):
    def __init__(self,num_layers,num_input_features,bn_size=4,growth_rate=16):
        super(_denseblock,self).__init__()
        for i in range(num_layers):
            layer=_denselayer(num_input_features+i*growth_rate,bn_size,growth_rate)
            self.add_module("denselayer%d" % (i+1,),layer)
            
    def forward(self,x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
    
def dice_loss(inputs, target):
    smooth = 1.

    iflat = inputs.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def mixup_data(x, segs, labels, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    segs_a, segs_b = segs, segs[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_x, segs_a, segs_b, labels_a, labels_b, lam

class Net2(nn.Module):
    def __init__(self,growth_rate=16,num_layers=4,num_init_features=16,bn_size=4,num_classes=2):
        super(Net2, self).__init__()
        self.pre=nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=3,stride=1,padding=1),
            #nn.BatchNorm3d(num_init_features),
            #nn.ReLU(inplace=True),
        )
        self.db1=_denseblock(
                num_layers=num_layers,
                num_input_features=num_init_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )        
        num_features=num_init_features+num_layers*growth_rate        
        mask_features1=num_features
        self.tr1=nn.Sequential(nn.Conv3d(num_features,num_features // 2,kernel_size=3,padding=1,bias=True),
                              nn.AvgPool3d(2, stride=2)
                              )       
        num_features = num_features//2  
        self.db2=_denseblock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )   
        num_features=num_features+num_layers*growth_rate  
        mask_features2=num_features
        self.tr2=nn.Sequential(nn.Conv3d(num_features,num_features // 2,kernel_size=3,padding=1,bias=True),
                              nn.AvgPool3d(2, stride=2)
                              )     
        num_features = num_features//2 
        self.db3=_denseblock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )   
        num_features=num_features+num_layers*growth_rate  
        mask_features3=num_features
        self.tr3=nn.Sequential(nn.Conv3d(num_features,num_features // 2,kernel_size=3,padding=1,bias=True),
                              nn.AvgPool3d(2, stride=2)
                              )       
        num_features = num_features//2  
        self.db4=_denseblock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )   
        num_features=num_features+num_layers*growth_rate  
        self.final = nn.Sequential(nn.Conv3d(num_features,num_features//2,kernel_size=3,stride=1,padding=1,bias=True),
                                  )
        num_features = num_features//2 
        self.fc=nn.Sequential(nn.Linear(num_features,num_features),
                              nn.ReLU(inplace=True),
                              nn.Dropout(),
                              nn.Linear(num_features,num_features),
                              nn.ReLU(inplace=True),
                              nn.Dropout(),
                             nn.Linear(num_features,num_classes))
        #self.fc=nn.Sequential(
         #                     nn.Linear(num_features,num_classes),
          #                   )
        self.maskconv1=nn.ConvTranspose3d(mask_features3,mask_features2,kernel_size=2,stride=2)
        self.maskconv2=nn.ConvTranspose3d(mask_features2,mask_features1,kernel_size=2,stride=2)
        self.maskconv3=nn.Conv3d(mask_features1,1,kernel_size=1,stride=1)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)
        
    def forward(self,x):
        y1=self.pre(x)
        #print(x.shape)
        y1=self.db1(y1)
        #print(y1.shape)
        y2=self.db2(self.tr1(y1))
        #print(y2.shape)
        y3=self.db3(self.tr2(y2))
        #print(y3.shape)
        out=self.db4(self.tr3(y3))
        #print(out.shape)
        out=self.final(out)
        out=F.adaptive_avg_pool3d(out, (1, 1, 1))
        out=torch.flatten(out, 1)
        out=self.fc(out)
        #out=F.softmax(out,dim=1)
        mask=torch.sigmoid(self.maskconv3(self.maskconv2(self.maskconv1(y3)+y2)+y1))
        return mask,out
    
class Net4(nn.Module):
    def __init__(self,growth_rate=16,num_layers=4,num_init_features=8,bn_size=4,num_classes=2):
        super(Net4, self).__init__()
        self.pre=nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=3,stride=1,padding=1),
            #nn.BatchNorm3d(num_init_features),
            #nn.ReLU(inplace=True),
        )
        self.pre2=nn.Conv3d(1, num_init_features, kernel_size=5,stride=1,padding=2)
        self.pre3=nn.Conv3d(1, num_init_features, kernel_size=7,stride=1,padding=3)
        num_init_features=num_init_features*3
        '''
        self.pre4=nn.Conv3d(num_init_features,num_init_features,kernel_size=3,stride=1,padding=1)
        '''
        self.db1=_denseblock(
                num_layers=num_layers,
                num_input_features=num_init_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )        
        num_features=num_init_features+num_layers*growth_rate        
        mask_features1=num_features
        self.tr1=nn.Sequential(nn.Conv3d(num_features,num_features // 2,kernel_size=3,padding=1,bias=True),
                              nn.AvgPool3d(2, stride=2)
                              )       
        num_features = num_features//2  
        self.db2=_denseblock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )   
        num_features=num_features+num_layers*growth_rate  
        mask_features2=num_features
        self.tr2=nn.Sequential(nn.Conv3d(num_features,num_features // 2,kernel_size=3,padding=1,bias=True),
                              nn.AvgPool3d(2, stride=2)
                              )     
        num_features = num_features//2 
        self.db3=_denseblock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )   
        num_features=num_features+num_layers*growth_rate  
        mask_features3=num_features
        self.tr3=nn.Sequential(nn.Conv3d(num_features,num_features // 2,kernel_size=3,padding=1,bias=True),
                              nn.AvgPool3d(2, stride=2)
                              )       
        num_features = num_features//2  
        self.db4=_denseblock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )   
        num_features=num_features+num_layers*growth_rate  
        mask_features4=num_features
        #print(num_features)
        self.final1 = nn.Conv3d(num_features,num_features//5,kernel_size=3,stride=1,padding=1,bias=True)
        self.final2 = nn.Conv3d(num_features,num_features//5,kernel_size=5,stride=1,padding=2,bias=True)
        self.final3 = nn.Conv3d(num_features,num_features//5,kernel_size=7,stride=1,padding=3,bias=True)
        
        num_features = (num_features//5)*3 
        self.fc=nn.Sequential(
            nn.Linear(num_features,num_features),
                              nn.ReLU(inplace=True),
                              nn.Dropout(),
                              #nn.Linear(num_features,num_features),
                              #nn.ReLU(inplace=True),
                              #nn.Dropout(),
                             nn.Linear(num_features,num_classes))
        #self.fc=nn.Sequential(
         #                     nn.Linear(num_features,num_classes),
          #                   )
        
        self.maskconv1=nn.ConvTranspose3d(mask_features4,mask_features3,kernel_size=2,stride=2)
        self.maskconv2=nn.ConvTranspose3d(mask_features3,mask_features2,kernel_size=2,stride=2)
        self.maskconv3=nn.ConvTranspose3d(mask_features2,mask_features1,kernel_size=2,stride=2)
        self.maskconv4=nn.Conv3d(mask_features1,1,kernel_size=1,stride=1)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)
        
    def forward(self,x):
        #print(x.shape)
        y1=torch.cat((self.pre(x),self.pre2(x),self.pre3(x)),1)
        #y1=self.pre4(y1)
        #print(y1.shape)
        y1=self.db1(y1)
        #print(y1.shape)
        y2=self.db2(self.tr1(y1))
        #print(y2.shape)
        y3=self.db3(self.tr2(y2))
        #print(y3.shape)
        y4=self.db4(self.tr3(y3))
        #print(out.shape)
        out=torch.cat((self.final1(y4),self.final2(y4),self.final3(y4)),1)
        out=F.adaptive_avg_pool3d(out, (1, 1, 1))
        out=torch.flatten(out, 1)
        out=self.fc(out)
        #out=F.softmax(out,dim=1)
        mask=torch.sigmoid(self.maskconv4(self.maskconv3(self.maskconv2(self.maskconv1(y4)+y3)+y2)+y1))
        return mask,out

#%%
net1=Net2(num_init_features=16,bn_size=4).to(device)
net1.load_state_dict(torch.load(model_dir))

#%%
f = open('submission.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(["name","predicted"])

test_dir=data_dir+'test/test/'
test_npz=os.listdir(test_dir)

net1.eval()
with torch.no_grad():
    for i in range(test_num):
        input_voxel=vox_transform(np.load(test_dir+test_npz[i])['voxel'][34:66,34:66,34:66]).to(device)
        _,predict = net1(input_voxel.view(1,1,32,32,32))
        predict = predict.cpu()
        csv_writer.writerow([test_npz[i][:-4],predict[0,1].item()])

f.close()

#%%


