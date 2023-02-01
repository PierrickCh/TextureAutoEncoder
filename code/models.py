import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torchvision.transforms.functional as TF
from torchvision import transforms

import config
from util import Normalize_gradients
I = Normalize_gradients().apply

model_dir = "/scratchm/pchatill/projects/comparison/vgg_conv.pth"



class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        if config.args.center_gram: # center feature maps before gram matrix correlation: default= yes
            F=F-F.mean(dim=-1,keepdim=True)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)  # only divides by spatial dim
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)



class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        """returns a gaussian pyramid of an image, used when texture_loss=='snelgrove' with a shallower VGG network used in a multiscale fashion"""
        super(GaussianSmoothing, self).__init__()
        # 2D Gaussian kernel parameters
        kernel_size = [kernel_size] * 2
        sigma = [sigma] * 2
        # Gaussian kernel
        kernel = 1
        mgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) \
                                 for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, mgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        # Repeat for 3 channel (batch) input
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        # Network layers
        self.conv = nn.Conv2d(channels, channels, kernel_size, groups=channels, bias=False, stride=2,
                              padding=int((kernel_size[0] - 1) / 2),padding_mode='replicate')
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, input, scales):
        l = []
        x = input
        l.append(input)
        for k in range(scales - 1):
            x = self.conv(x)
            l.append(x)
        return l




class Quad_module(nn.Module):
    """Quadratic feature extractor, forward takes input of size B,nc_in,H,W 
    and outputs a learned representation of size B,nc_out.
    Parameters:
        nc_in  : number of feature maps of the input features
        nc_out : size of the quadratic representation
    """
    def __init__(self, n_in, n_out):
        super(Quad_module, self).__init__()
        self.grad_boost = 10
        self.w1 = nn.Parameter(5 / self.grad_boost * torch.rand(n_in, n_out, requires_grad=True))
        self.softmax = nn.Softmax(dim=0)
        self.n_out = n_out
        self.n_in = n_in

    def forward(self, x):
        N, C, H, W = x.shape
        M = x.view(N, C, H * W)
        out = torch.matmul(M.transpose(1, 2), self.softmax(self.w1 * self.grad_boost)) ** 2
        return torch.sum(out, dim=1)



class Norm(nn.Module):
    """Normalizes each channel independently of the others and of the other elements of the batch.
    Is used in the encoder due to small batch size suring training
    """
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        m, s = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        return (x - m) / (s + 1e-8)


class Texture_Encoder(nn.Module):
    """Texture encoder, forward takes as input a list of 
    VGG features taken at different stages of the network
    (with number of channels listed in list_channels),
    outputs a texture descriptor of size nc_w.
    nc_quad : size of the quadratic descriptor for each feature volume"""
    def __init__(self, list_channels=[64, 128, 256, 512, 512], n_quad=128, nc_w=128):
        super(Texture_Encoder, self).__init__()
        self.nc_w=nc_w
        self.quad_list = nn.ModuleList([])
        self.norm=nn.ModuleList([])
        for nc in list_channels:
            self.norm.append(Norm())
            self.quad_list.append(Quad_module(nc, n_quad).cuda())
        self.fc = nn.Sequential(nn.Linear(len(list_channels) * n_quad, 512), Norm(), nn.LeakyReLU(),
                            nn.Linear(512, 512), Norm(), nn.LeakyReLU(), nn.Linear(512, nc_w))
        
    def forward(self, vgg_outs):
        l = []
        for quad, norm, out in zip(self.quad_list, self.norm, vgg_outs):
            q = norm(quad(out))
            l.append(q)
        l = torch.cat(l, dim=1)
        out=self.fc(l)
        return out




class AdaIN(nn.Module):
    def __init__(self):
        """code modified from https://github.com/CellEight/Pytorch-Adaptive-Instance-Normalization"""
        super().__init__()
        self.width = 4

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w)"""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor"""
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) + 0.000000023) / (
                        x.shape[2] * x.shape[3]))

    def forward(self, x, mu, sigma):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style."""

        return (sigma * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + mu).permute([2, 3, 0, 1])

    def forward_map(self, x, mu, sigma):
        """Performs the forward pass with mu and sigma being a 2D map instead of scalar.
        Each location is normalized by local statistics, and denormalized by mu and sigma evaluated at the given location.
        The local statistics are computed with a gaussian kernel of relative width controlled by the variable local_stats_width defined at the top of this file"""
        B, C, H0, W0 = x.shape
        pool=nn.AdaptiveAvgPool2d((min(H0,128),min(W0,128))) # the pooling operation is used to make the computation of local mu and sigma less expensive
                                                             # using the spacial smoothness of locally computed statistics
        x_pooled=pool(x) # x_pooled of maximum spacial size 128*128

        #n ow create the gaussian kernel for loacl stats computation
        B, C, H, W = x_pooled.shape
        rx,ry=H0/H,W0/W
        width=self.width/min(rx,ry)
        kernel_size = [(max(int(2 * width), 5) // 2) * 2 + 1, (max(int(2 * width), 5) // 2) * 2 + 1] # kernel size is odd
        width = [width,width]
        kernel = 1
        mgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, width, mgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel_1d = kernel.view(1, 1, *kernel.size())
        kernel = kernel_1d.repeat(C, *[1] * (kernel_1d.dim() - 1)).cuda()

        # create a weight map by convolution of a constant map with the gaussian kernel. It used to correctly compute the local statistics at the border of the image, accounting for zero padding
        ones=torch.ones(1,1,H,W).cuda()
        weight=F.conv2d(ones,kernel_1d.cuda(),bias=None,padding='same')

        # define channel-wise gaussian convolution module conv
        conv = nn.Conv2d(C, C, kernel_size, groups=C, bias=False, stride=1, padding=int((kernel_size[0] - 1) / 2),
                         padding_mode='zeros')
        conv.weight.data = kernel
        conv.weight.requires_grad = False
        

        local_mu = conv(x_pooled)/weight # pooling already performs local averaging, so it does not perturb the computation of the local mean
        local_mu=F.interpolate(local_mu, size=(H0,W0), mode='bilinear', align_corners=False) # now upsample the local mean map to the original shape

        local_sigma = torch.sqrt(conv(pool(((x-local_mu)**2)) /weight) + 10 ** -8) # perform (x-local_mu)**2 at the high resolution, THEN pool and finally smooth to get the local standard deviation.
        
        local_sigma=F.interpolate(local_sigma, size=(H0,W0), mode='bilinear', align_corners=False) # upsample the local std map to the original shape
    
        #finally perform the local AdaIN operation using these maps of local_mu and local_sigma to normalize, then denormalize with the given maps mu and sigma.
        x_norm=(x - local_mu) / local_sigma
        return (sigma * x_norm + mu)



class PeriodicityUnit(nn.Module):
    """Module used to add periodic information to a feature map.
    Given frequencies of x coordinates fx (B,n_freq) and y coordinates (B,n_freq)
    input x (B,nc_in,H,W)
    output (B,nc_out,H,W)
    """
    def __init__(self, in_channels, out_channels,nc_w,  n_freq=0):
        super(PeriodicityUnit, self).__init__()

        self.n_freq = n_freq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ada=AdaIN()


        self.sine_maps_conv = nn.Conv2d(n_freq, out_channels, 1)
        self.to_style = nn.Linear(nc_w, n_freq)

        

    def forward(self, x, w_mod,w_map=None, fx=None, fy=None, phase=None):
        B,C,H,W = x.size()

        w_grid = torch.arange(start=0, end=W).repeat(H, 1).view(1, H, W) - (W - 1) / 2
        h_grid = torch.arange(start=0, end=H).repeat(W, 1).T.view(1, H, W) - (H - 1) / 2
        grid = torch.cat([w_grid, h_grid]).view(2, -1).type(torch.cuda.FloatTensor)
        random_phase = 2 * math.pi * torch.rand(1).to("cuda")
        phase = random_phase if phase is None else phase
        freq = torch.stack((fy, fx), dim=2)
        r = torch.sqrt(fx ** 2 + fy ** 2)

        if config.args.freq_amp == "2scales": #default: a frequency is used in the 2 levels where it appears as a not too high nor too low
            freq_amp = torch.maximum(torch.tensor(0.).cuda(), 1 - torch.abs(1 - torch.log2(
                8 * r)))  
        elif config.args.freq_amp == "trans_scales":
            freq_amp = torch.minimum(torch.tensor(1.).cuda(), torch.maximum(torch.tensor(0.).cuda(), -torch.log2(
                8 * r)))  

        if w_map is None:
            sines = freq_amp.unsqueeze(-1) * torch.sin(torch.matmul(2 * math.pi * freq, grid) + phase) # sine maps modulated accordingly to their magnitude
            modulation = self.to_style(w_mod) # input-specific modulation
            sines = sines.view(B, self.n_freq, H, W) * modulation.unsqueeze(-1).unsqueeze(-1)
            out = self.sine_maps_conv(sines)
        else:
            dh,dw=x.shape[2]-w_map.shape[2],x.shape[3]-w_map.shape[3]
            w_map=F.pad(w_map,(dw//2,dw-dw//2,dh//2,dh-dh//2),mode='replicate')
            fx=F.pad(fx,(dw//2,dw-dw//2,dh//2,dh-dh//2),mode='replicate')
            fy=F.pad(fy,(dw//2,dw-dw//2,dh//2,dh-dh//2),mode='replicate')
            freq = torch.stack((fy, fx), dim=2)
            r = torch.sqrt(fx ** 2 + fy ** 2)
            if config.args.freq_amp == "2scales":
                freq_amp = torch.maximum(torch.tensor(0.).cuda(), 1 - torch.abs(1 - torch.log2(8 * r))) 
            elif config.args.freq_amp == "trans_scales":
                freq_amp = torch.minimum(torch.tensor(1.).cuda(), torch.maximum(torch.tensor(0.).cuda(), -torch.log2(8 * r)))  
            sines = freq_amp.view(B,self.n_freq,-1)* torch.sin((2 * math.pi * freq.view(B,self.n_freq,2,-1)*grid.unsqueeze(0).unsqueeze(0)).sum(-2) + phase)
            modulation = self.to_style(w_map.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            sines = sines.view(B, self.n_freq, H, W) * modulation
            out = self.sine_maps_conv(sines)  
        return out, modulation.abs().mean()






class conv(nn.Module):
    """"convolution layer to use across layers, useful to control padding mode"""
    def __init__(self, n_ch_in, n_ch_out, k=3):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(n_ch_in, n_ch_out, k, padding=0)  # padding=k//2,padding_mode='circular')
    def forward(self, x):
        return self.conv(x)




class Z_to_w(nn.Module):
    def __init__(self, nc_z,nc_w,depth=10,nc_latent=256):
        """T network in the paper, used for direct sampling of the generator. Learns a meaningful mapping form a gaussian noise to the latent space W."""
        super(Z_to_w, self).__init__()
        l=[]
        l += [nn.Linear(nc_z, nc_latent), nn.LeakyReLU(.1), nn.BatchNorm1d(nc_latent)]
        for _ in range(depth-2):
            l += [nn.Linear(nc_latent, nc_latent), nn.LeakyReLU(.1), nn.BatchNorm1d(nc_latent)]
        l += [nn.Linear(nc_latent, nc_w)]
        self.fc=nn.Sequential(*l)
    def forward(self,z):
        return self.fc(z)





class W_Discriminator(nn.Module):
    def __init__(self, nc_w=128):
        """Discriminator used to train T (Z_to_w in the code). Operates in the latent space W"""
        super().__init__()
        self.fc = [nn.Linear(nc_w, 256), nn.LeakyReLU(),  # 182
                   nn.Linear(256, 256), nn.LeakyReLU(),
                   nn.Linear(256, 256), nn.LeakyReLU(),
                   nn.Linear(256, 1)]
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        out = self.fc(x)
        return out



class Pred(nn.Module):
    def __init__(self, nc_w=128):
        """Prediction ntework. Takes as input a batch of latent variables w and outputs a scale and rotation parameter for each element"""
        super().__init__()
        self.fc = [nn.Linear(nc_w, 512), Norm(), nn.LeakyReLU(),
                   nn.Linear(512, 128), Norm(), nn.LeakyReLU(),
                   nn.Linear(128, 3)]
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        out = self.fc(x)
        logit_scale = out[...,0]# predict a log scale proved to be more flexible
        scale = 2 ** (logit_scale - 1)
        x, y = out[..., 1], out[..., 2]# instead of predicting an angle directly, we avoid periodicity complications by predicting a point in the 2D plane, and taking its argument
        theta = torch.atan2(y,x) / 2 # divide by 2 to get pi-periodic result, as orientation of sine waves is pi-periodic
        return scale, theta







class StyleBlock(nn.Module):
    def __init__(self, nc_in, nc_out, w_dim, n_freq=0):
        """core block of the generator, similarly to StyleGAN blocks:
        uses the latent variable w to predict mu and sigma used in AdaIN
        perfomrs convolution 3*3 and addition of noise
        alose adds periodic content
        """
        super().__init__()
        self.n_freq = n_freq
        self.nc_in=nc_in
        self.nc_out=nc_out
        
        if config.args.sine_maps:
            if config.args.sine_maps_merge == 'add':
                self.periodic_content = PeriodicityUnit(nc_in, nc_out, w_dim, n_freq)
                self.conv1 = conv(nc_in, nc_out)
            else: #if sine_maps_merge != 'add', then perform concatenation instead
                self.periodic_content = PeriodicityUnit(nc_in, nc_out // 2, w_dim,n_freq)
                self.conv1 = conv(nc_in, nc_out // 2)
        else: #no sine maps used
            self.periodic_content=None
            self.conv1 = conv(nc_in, nc_out)

        self.conv2 = conv(nc_out, nc_out)

        self.nl = nn.LeakyReLU(.1)
        self.ada = AdaIN()
        
        self.to_style1 = nn.Linear(w_dim, nc_out * 2)
        self.to_style2 = nn.Linear(w_dim, nc_out * 2)  
        
        self.n1, self.n2 = None, None  # noise maps to save for inference experiments with fixed spacial realization, such as interpolation in the latent space
        self.noise_modulation1 = Parameter(.01 * torch.randn(1, nc_out, 1, 1).cuda(), requires_grad=True) # learnable modulation of the noise
        self.noise_modulation2 = Parameter(.01 * torch.randn(1, nc_out, 1, 1).cuda(), requires_grad=True) 
    
    
    def forward(self, x, w, fx, fy, w_map=None, save_noise=False, phase=None):

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x_conv = self.conv1(x)


        if self.periodic_content is not None: # retrieve periodic content and merge it to the current feature maps accordingly to sine_maps_merge mode
            x_f_conv, l1_loss = self.periodic_content(x, w, w_map, fx, fy, phase=phase)  
            x_f_conv = x_f_conv[..., 1:-1, 1:-1]
            self.l1_loss = l1_loss
            if config.args.sine_maps_merge == 'add': # addition
                x = x_conv + x_f_conv
            else: # concatenation
                x = torch.cat((x_conv, x_f_conv), dim=1)
        else:
            self.l1_loss=0.*w.mean() #not breaking anything in the logs
            x=x_conv #not changing anything
      

        if save_noise and self.n1 is None: # deal with saving, reusing, or directly sampling noise map
            self.n1 = torch.randn(1,*x.shape[1:]).cuda()
            n1 = self.n1
        elif save_noise:
            n1 = self.n1
        else:
            self.n1 = None
            n1 = torch.randn(x.shape).cuda()
        

        if w_map is None: #core of a style block: convolution, noise addition and AdaIN module with predicted mu and sigma
            style = self.to_style1(w)
            mu, sigma = style[..., :self.nc_out], style[..., self.nc_out:self.nc_out * 2]
            x += n1 *self.noise_modulation1 
            x = self.ada(x, mu, sigma) 
        else:
            dh,dw=x.shape[2]-w_map.shape[2],x.shape[3]-w_map.shape[3]
            w_map=F.pad(w_map,(dw//2,dw-dw//2,dh//2,dh-dh//2),mode='replicate') #w_map is a small map of differnent textures style in different locations.
                                                                                # It needs to be smoothly upsampled to the size of the current feature map
            style_map = self.to_style1(w_map.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            mu, sigma = style_map[:, :self.nc_out], style_map[:, self.nc_out:2 * self.nc_out]
            x+=n1 * self.noise_modulation1
            x = self.ada.forward_map(x, mu, sigma)


        style = self.to_style2(w)
        mu, sigma = style[:, :self.nc_out], style[:, self.nc_out:2 * self.nc_out]
        x = self.conv2(x)


        if save_noise and self.n2 is None:
            self.n2 = torch.randn(1,*x.shape[1:]).cuda()
            n2 = self.n2
        elif save_noise:
            n2 = self.n2
        else:
            self.n2 = None
            n2 = torch.randn(x.shape).cuda()
            

        if w_map is None:
            style = self.to_style2(w)
            mu, sigma = style[..., :self.nc_out], style[..., self.nc_out:self.nc_out * 2]
            x += n2  *self.noise_modulation2 #* modulation_noise.unsqueeze(-1).unsqueeze(-1)*0.01
            x = self.ada(x, mu, sigma)
        else:
            dh,dw=x.shape[2]-w_map.shape[2],x.shape[3]-w_map.shape[3]
            w_map=F.pad(w_map,(dw//2,dw-dw//2,dh//2,dh-dh//2),mode='replicate')
            style_map = self.to_style2(w_map.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            mu, sigma = style_map[:, :self.nc_out], style_map[:, self.nc_out:2 * self.nc_out]
            x += n2  * self.noise_modulation2
            x = self.ada.forward_map(x, mu, sigma)

        x = self.nl(x) # apply non-linearity was found essential expermentally

        return x


class StyleConv(nn.Module):
    def __init__(self, nc_in, nc_out, w_dim, k=3):
        """basic styleblock operation: convolution followed by AdaIN.
        Used at the end of the generator to project to RGB"""
        super().__init__()
        self.conv1 = conv(nc_in, nc_out, k)
        self.ada = AdaIN()
        self.nc_out = nc_out
        self.to_style = nn.Linear(w_dim, nc_out * 2)
    def forward(self, x, w, w_map=None):
        if w_map is None:
            style = self.to_style(w)  # w is B,nc_w
            mu, sigma = style[..., :self.nc_out], style[..., self.nc_out:]
            x = self.conv1(x)
            x = self.ada(x, mu, sigma)
        else:
            x = self.conv1(x)
            dh,dw=x.shape[2]-w_map.shape[2],x.shape[3]-w_map.shape[3]
            w_map=F.pad(w_map,(dw//2,dw-dw//2,dh//2,dh-dh//2),mode='replicate')
            style_map = self.to_style(w_map.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            mu, sigma = style_map[:, :self.nc_out], style_map[:, self.nc_out:2 * self.nc_out]
            x = self.ada.forward_map(x, mu, sigma)
        return x
















class style_generator(nn.Module):
    def __init__(self, nc_w=128, n_freq=0,n=7):
        '''Network G of the paper'''
        super(style_generator, self).__init__()
        
        self.zoom=(1,1)
        self.n_freq = n_freq

        nc_max = config.args.nc_max  #argument controlling the maximal depth of features maps in the network.
        self.input_tensor = Parameter(torch.randn(1, nc_max, 1, 1).cuda(), requires_grad=True) #not 4*4 like in StyleGAN, but a constant replicated along spacial axes at the start of the network

        self.n = n #number of levels in the generator
        self.pad = 4  #the input tensor is expanded to a greater spacial size to account for the fact that no padding is performed in any convolution in the generator
        
        # for inference purposes
        self.save_noise = False
        self.offset= None 

        self.pred = Pred(nc_w) #scale and rotation predictor network

        self.scale, self.theta = None, None # for plotting histograms of predicted scale and rotation parameters

        self.grad_boost = 100 # learnable frequency are learned with a greater learning rate with a factor 100
        if n_freq != 0:# initialization of learnable frequencies
            self.r = nn.Parameter((torch.linspace(1, self.n, n_freq) / self.grad_boost).cuda(), requires_grad=True) # log magnitude of the frequencies, will be exponentiated to get the magnitude
            self.phase = nn.Parameter(
                (torch.linspace(0, np.pi, n_freq)[torch.randperm(n_freq)] / self.grad_boost).cuda(), requires_grad=True)
        
        
        l = []
        for i in range(self.n): # main body: cascade of StyleBlock modules 
            l.append(StyleBlock(min(nc_max, 2 ** (5 + self.n - i)), min(nc_max, 2 ** (5 + self.n - i - 1)), nc_w, n_freq=n_freq))
        self.body_modules = nn.ModuleList(l)

        def body_forward(x, w, w_map=None):
            if w_map is None:
                s, t = self.pred(w) # infer scale and rotation
                mod = 2 ** (-self.r * self.grad_boost) # get magnitude of each frequency
                fx, fy = s.unsqueeze(1) * mod.unsqueeze(0) * torch.cos(
                    self.phase.unsqueeze(0) * self.grad_boost + t.unsqueeze(1)), s.unsqueeze(1) * mod.unsqueeze(
                    0) * torch.sin(self.phase.unsqueeze(0) * self.grad_boost + t.unsqueeze(1))  #  In this operation, we retrieve the cartesian coordonates of each transformed frequency f'_i
                #  the first two dimensions of fx are (batch_size,n_freq,...) each frequency is transformed differently according to the element of the batch through the latent variable w,
                #  that yields exemplar-specific scale and rotation prediction.
                #  For each element in the batch b, all the learned frequencies are scaled and rotated with the same predicted parameters s_b and t_b.
    
                self.scale, self.theta = s, t # for logging
            
            if self.save_noise and self.offset is None: # for inference experiments, phase is random durin training
                self.offset=2 * math.pi * torch.rand(1, 1, 1).to("cuda")
                offset=self.offset
            elif self.save_noise:
                offset=self.offset
            else:
                offset = 2 * math.pi * torch.rand(x.shape[0], 1, 1).to("cuda")

            for i, m in enumerate(self.body_modules):
                if w_map is not None:
                    w_map_curr=F.interpolate(w_map, (2**(i+2)*self.zoom[0],2**(i+2)*self.zoom[1]), mode='bilinear', align_corners=True)
                    m.ada.width=2**(i+2)*config.args.local_stats_width*min(self.zoom[0],self.zoom[1]) #important detail: local_stats_width controls how local stats are computed you may try from .1 to .8
                    s, t = self.pred(w_map_curr.permute((0, 2, 3, 1)))
                    if config.args.sine_maps:
                        mod = 2 ** (-self.r * self.grad_boost)
                        fx, fy = s.unsqueeze(1) * mod.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * torch.cos(
                            self.phase.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * self.grad_boost + t.unsqueeze(1)), s.unsqueeze(1) * mod.unsqueeze(
                            0).unsqueeze(-1).unsqueeze(-1) * torch.sin(self.phase.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * self.grad_boost + t.unsqueeze(1)) 
                    else:
                        fx, fy = 0, 0
                    x = m(x, w, fx * 2 ** (self.n - i - 1), fy * 2 ** (self.n - i - 1), w_map=w_map_curr, save_noise=self.save_noise,
                      phase=offset * 2 ** -i)
                else:
                    x = m(x, w, fx * 2 ** (self.n - i - 1), fy * 2 ** (self.n - i - 1), w_map=w_map, save_noise=self.save_noise,
                      phase=offset * 2 ** -i) #phases are scaled in order for the sine waves to be aligned in the two consectutive leveks in which a frequency is used
            return x

        self.body = body_forward
        self.rgb =StyleConv(32, 3, nc_w,k=1)

    def forward(self,w, w_map=None,zoom=(1,1)):
        self.zoom=zoom
        self.pad=4 if zoom==(1,1) else 5

        x = self.body(self.input_tensor.repeat(w.shape[0], 1, 2**(8-self.n)*zoom[0] + self.pad, 2**(8-self.n)*zoom[1] + self.pad), w, w_map)

        if w_map is not None:
            self.rgb.ada.width=2**(self.n+1)*config.args.local_stats_width*min(self.zoom[0],self.zoom[1])
            x = self.rgb(x, w,  w_map=F.interpolate(w_map, size=(2**(self.n+1)*zoom[0],2**(self.n+1)*zoom[1]), mode='bilinear', align_corners=True))
        else:
            x = self.rgb(x, w, w_map)
        
        if self.training:
            x=transforms.RandomCrop((256*zoom[0],256*zoom[1]))(x)
        else:
            x = TF.center_crop(x, (256*zoom[0],256*zoom[1]))
        
        x = torch.tanh(x)  
        return x


