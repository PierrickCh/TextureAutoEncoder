from sys import stderr
import torch
import PIL.Image as Image
import numpy as np
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import json
import sys
import config




def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


prep = transforms.Compose(
[transforms.Lambda(lambda x: x[:, torch.LongTensor([2, 1, 0])]),  # turn to BGR
transforms.Normalize(-1, 2),
transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
std=[1, 1, 1]),
transforms.Lambda(lambda x: x.mul_(255)),])


MEAN = (0.485, 0.456, 0.406)
mean = torch.as_tensor(MEAN).view(-1, 1, 1).cuda()

def prep(x):
    x=x/2+.5
    return x.sub_(mean).mul_(255)


class Ds_folder(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.l = []
        for f in os.listdir(dataset_path):
            
            self.l.append(os.path.join(dataset_path, f))

        self.transform = transform

    def __len__(self):
        return len(self.l)

    def __getitem__(self, idx):
        if self.transform:
            img = Image.open(self.l[idx]).convert('RGB')
            img = self.transform(img)
        return img




def get_loader():
    #TODO change paths
    """prepares the dataloader to use during training"""
    
    torch.manual_seed(config.args.seed)


    transformations = transforms.Compose([#transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1),
    transforms.RandomResizedCrop((400, 400), (1, 1), (1, 1)), transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))])

    dset = Ds_folder(config.args.dataset_folder, transformations)
    test_size=32
    train_dset, test_dset = torch.utils.data.random_split(dset, [len(dset)-test_size,test_size],generator=torch.Generator().manual_seed(42)) # change seed for different train/test split

    config.args.test_split=[dset.l[i] for i in test_dset.indices] #saves paths of chosen test data
    loader = DataLoader(train_dset, batch_size=config.args.batch_size, shuffle=True, drop_last=True,num_workers=4)
    loader_test = DataLoader(test_dset, batch_size=config.args.batch_size, shuffle=True, drop_last=False,num_workers=4)
    return loader,loader_test



class Normalize_gradients(Function):

    @staticmethod
    def forward(self, input):
        return input.clone()

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        if grad_input.dim()==4:
            m = grad_input.norm(p=2, keepdim=True, dim=1).norm(p=2, keepdim=True, dim=2).norm(p=2, keepdim=True, dim=3)
            if m.sum()!=0:
                grad_input = grad_input.mul(1. / m)
        elif grad_input.dim()==3:
            m = grad_input.norm(p=2, keepdim=True, dim=1).norm(p=2, keepdim=True, dim=2)
            if m.sum()!=0:
                grad_input = grad_input.mul(1. / m)
        elif grad_input.dim()==2:
            m = grad_input.norm(p=2, keepdim=True, dim=1)
            if m.sum()!=0:
                grad_input = grad_input.mul(1. / m)
        
        return grad_input,




def calc_gradient_penalty(netD, real_data, fake_data, device='cuda:0'):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)  # .cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    outs = netD(interpolates)
    disc_interpolates = torch.cat([out.view(out.shape[0], -1) for out in outs], dim=-1)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def augment(real,scale,theta):
    aff = torch.stack(
        (torch.stack((torch.cos(theta) * scale, -torch.sin(theta) * scale, 0 * scale), dim=1),
        torch.stack((torch.sin(theta) * scale, torch.cos(theta) * scale, 0 * scale), dim=1)),
        dim=1)
    grid = F.affine_grid(aff, real.size())
    t_real = F.grid_sample(real, grid)
    t_real = TF.center_crop(t_real, 256)
    return t_real


def angle(a1, a2):
    return (1 - torch.cos(2 * (a1 - a2))) / 2

def anglel1(a1, a2):
    return 1 - 2 / np.pi * torch.abs(np.pi / 2 - torch.remainder(a1 - a2, torch.tensor(np.pi)))

def inv(perm):
    _, ind = torch.sort(perm)
    return ind


def Hist_loss(im,img):
    loss=0.
    b, c, h, w = im.shape

    for i in range(10):
        v=torch.randn(3,1,dtype=im.dtype).cuda()
        v=v/v.norm()

        im_proj = torch.matmul(im.reshape(b, c, -1).transpose(1, 2), v)
        gt_proj = torch.matmul(img.reshape(b, c, -1).transpose(1, 2), v)
        sorted, indices = torch.sort(im_proj[..., 0])
        sorted_gt, indices_gt = torch.sort(gt_proj[..., 0])

        stretched_proj = F.interpolate(sorted_gt.unsqueeze(1), size=indices.shape[-1],mode = 'nearest', recompute_scale_factor = False) # handles a generated image larger than the ground truth 
        diff = (stretched_proj[:, 0] - sorted) #[inv(indices)]
        loss += .5*torch.mean(diff**2)

        
    return loss/10
