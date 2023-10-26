from sys import stderr
import torch
import PIL.Image as Image
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
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
std= [0.229, 0.224, 0.225])])
#,transforms.Lambda(lambda x: x.mul_(255)),])


MEAN = (0.485, 0.456, 0.406)
STD=(0.229, 0.224, 0.225)
mean = torch.as_tensor(MEAN).view(-1, 1, 1).cuda()
std = torch.as_tensor(STD).view(-1, 1, 1).cuda()

def prep(x):
    x=x/2+.5
    #return x.sub_(mean).div_(std)
    return x.sub_(mean).mul_(255)


class Ds_folder(Dataset):
    def __init__(self, dataset_path, transform=None,test_set=None,mode=None):
        self.l = []
        print(os.getcwd())
        print(dataset_path)
        if mode is None:
            for f in os.listdir(dataset_path):
                self.l.append(os.path.join(dataset_path, f))
        else:
            test_set_split=[e.split('/')[-1] for e in test_set]
            if mode=='test':
                self.l= [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
                self.l= [e for e in self.l if e.split('/')[-1]  in test_set_split]
            elif mode=='train':
                self.l= [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)] 
                self.l= [e for e in self.l if e.split('/')[-1] not in test_set_split]
            else:
                raise ValueError("wrong mode")
        self.l.sort() 

        self.transform = transform

    def __len__(self):
        return len(self.l)

    def __getitem__(self, idx):
        if self.transform:
            img = Image.open(self.l[idx]).convert('RGB')
            img = self.transform(img)
        return img





def get_loader(shuffle=True):
    #TODO change paths
    """prepares the dataloader to use during training"""

    if config.args.color_augmentation:
        transformations = transforms.Compose([transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.5),
        transforms.RandomResizedCrop((400, 400), (1, 1), (1, 1)), transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])       
    else:
        transformations = transforms.Compose([#transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.5),
        transforms.RandomResizedCrop((400, 400), (1,1), (1, 1)), transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])#!!

    
    try:
        test_dset=Ds_folder(config.args.dataset_folder, transformations,test_set=config.args.test_split,mode='test')
        train_dset=Ds_folder(config.args.dataset_folder, transformations,test_set=config.args.test_split,mode='train')
        loader = DataLoader(train_dset, batch_size=config.args.batch_size, shuffle=shuffle, drop_last=True,num_workers=4)
        loader_test = DataLoader(test_dset, batch_size=config.args.batch_size, shuffle=False, drop_last=False,num_workers=4)


    except:
        dset = Ds_folder(config.args.dataset_folder, transformations)
        test_size=32
        train_dset, test_dset = torch.utils.data.random_split(dset, [len(dset)-test_size,test_size],generator=torch.Generator().manual_seed(42)) # change seed for different train/test split
        config.args.test_split=[dset.l[i] for i in test_dset.indices] #saves paths of chosen test data
        #define the test split above, with a fixed external seed
        
        test_dset=Ds_folder(config.args.dataset_folder, transformations,test_set=config.args.test_split,mode='test')
        train_dset=Ds_folder(config.args.dataset_folder, transformations,test_set=config.args.test_split,mode='train')
        loader = DataLoader(train_dset, batch_size=config.args.batch_size, shuffle=shuffle, drop_last=True,num_workers=4)
        loader_test = DataLoader(test_dset, batch_size=config.args.batch_size, shuffle=False, drop_last=False,num_workers=4)
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

'''

def get_matrices(scale1, scale2, shear, theta):
    rot_mat = torch.stack(
        (torch.stack((torch.cos(theta), -torch.sin(theta)), dim=1),
         torch.stack((torch.sin(theta), torch.cos(theta)), dim=1)), dim=1)
    scale_mat = torch.stack(
        (torch.stack((scale1, 0. * scale1), dim=1),
         torch.stack((0 * scale1, scale2), dim=1)), dim=1)
    shear_mat = torch.stack(
        (torch.stack((1 + 0. * shear, shear), dim=1),
         torch.stack((0. * shear, 1 + 0. * shear), dim=1)), dim=1)
    aff = rot_mat @ shear_mat @ scale_mat

    scale1, scale2 = 1 / scale1, 1 / scale2
    shear = -shear
    theta = -theta
    rot_mat = torch.stack(
        (torch.stack((torch.cos(theta), -torch.sin(theta)), dim=1),
         torch.stack((torch.sin(theta), torch.cos(theta)), dim=1)), dim=1)
    scale_mat = torch.stack(
        (torch.stack((scale1, 0. * scale1), dim=1),
         torch.stack((0 * scale1, scale2), dim=1)), dim=1)
    shear_mat = torch.stack(
        (torch.stack((1 + 0. * shear, shear), dim=1),
         torch.stack((0. * shear, 1 + 0. * shear), dim=1)), dim=1)
    inv_aff = scale_mat @ shear_mat @ rot_mat
    return torch.cat((aff, 0. * aff[:, :, :1]), dim=2), torch.cat((inv_aff, 0. * inv_aff[:, :, :1]), dim=2)

def Aff_loss():
    scale1 = Variable((torch.rand(batch_size) * (max_scale - min_scale) + min_scale).cuda(),requires_grad=False)
    scale2 = Variable((torch.rand(batch_size) * (max_scale - min_scale) + min_scale).cuda(),requires_grad=False)
    theta = Variable((torch.rand(batch_size) * (max_theta - min_theta) + min_theta).cuda(), requires_grad=False)
    shear = Variable((torch.rand(batch_size) * (max_shear - min_shear) + min_shear).cuda(), requires_grad=False)

    same = diffusion_images[:1].repeat(batch_size, 1, 1, 1)
    same = augment(same, scale1,scale2,shear,theta) #  augmented with random rota
    diffusion_images = augment(diffusion_images, scale1,scale2,shear,theta) #  augmented with random rotation and scaling.



    t = torch.randint(0, diffusion.T, size=(images.shape[0],))
    y = diffusion.forward(diffusion_images, t)

    # Get the w from the texture encoder
    w = texture_encoder(diffusion_images)
    
    #loss aff_pred trains both Encoder and affine matrix prediction in the attention layer
    aff_pred = att_layer.affine_pred(w)[:,:,:2] #drop last colulm used for position bias in affine_grid
    inv_aff_pred0=torch.invers(aff[:1])
    aff,inv_aff=get_matrices(scale1,scale2,shear,theta) #not efficient, affine matrices alredy computed by the funtion augment
    aff,inv_aff=aff[:,:,:2],inv_aff[:,:,:2]
    loss_aff = torch.mean((inv_aff_pred0@aff_pred-inv_aff[:1]@aff)**2)
    return loss_aff

class Affine_pred(nn.Module):
    def __init__(self, nc_w=32):
        """ """
        super().__init__()
        self.fc = [nn.Linear(nc_w, 512), Norm(), nn.LeakyReLU(),
                   nn.Linear(512, 128), Norm(), nn.LeakyReLU(),
                   nn.Linear(128, 5)]
        self.fc = nn.Sequential(*self.fc)

    def forward(self, w):
        out = self.fc(w)
        logit_scale1, logit_scale2 = out[..., 0], out[..., 1]  # predict a log scale proved to be more flexible
        scale1, scale2 = 2 ** (logit_scale1 - 1), 2 ** (logit_scale2 - 1)
        x, y = out[..., 2], out[
            ..., 3]  # instead of predicting an angle directly, we avoid periodicity complications by predicting a point in the 2D plane, and taking its argument
        theta = torch.atan2(y,
                            x) / 2  # divide by 2 to get pi-periodic result, as orientation of sine waves is pi-periodic
        shear = F.softplus(out[..., 4])
        return scale1, scale2, theta, shear'''