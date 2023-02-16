
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import os
import torch.utils.data
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid,draw_bounding_boxes
import argparse
import PIL.Image as Image
import matplotlib
from torchvision.utils import save_image
import imageio
matplotlib.use('Agg')
from util import *
from models import *
import config
import imageio
from config import load_args
topil=transforms.ToPILImage()


if __name__ == '__main__':
    

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str, default='test',help='name of your experiment, as seen in the run folder')
    parser.add_argument('--text',type=str, default='yourtext')
    parser.add_argument('--n_gif',type=int, default=50,help='lenght of gifs animations')
    parser.add_argument('--bs',type=int, default=8,help='batch size for making interpolation gifs, reduce in case of memory issues')
    args_inference=parser.parse_args()




    assert args_inference.text.isalnum(), '--text must be alphanumeric'
    dir=os.path.join('./runs',args_inference.name)
    if not os.path.exists(dir):
        raise SystemExit('%s is not a directory with trained models'%dir)
    
    dir_exp=os.path.join(dir,'inference')
    if not os.path.exists(dir_exp):
        os.makedirs(dir_exp)

    args=config.get_arguments()
    args=load_args(os.path.join(dir,'arguments.json'),args)
    config.args=args
    torch.random.seed()


    vgg = models.vgg19(pretrained=False).features.cuda()
    pretrained_dict = torch.load('/scratchm/pchatill/projects/texture_comp/vgg_conv.pth')
    for param, item in zip(vgg.parameters(), pretrained_dict.keys()): 
        param.data = pretrained_dict[item].type(torch.FloatTensor).cuda()
    vgg.requires_grad_(False)
    outputs = {}
    def save_output(name):
        def hook(module, module_in, module_out):
            outputs[name] = module_out
        return hook
    layers = [1, 6, 11, 20, 29]
    layers_weights = [1/n**2 for n in [64,128,256,512,512]]
    for layer in layers:
        handle = vgg[layer].register_forward_hook(save_output(layer))

    z_to_w=Z_to_w(nc_z=args.nc_z, nc_w=args.nc_w,depth=args.depth_T).cuda()
    G = style_generator(nc_w=args.nc_w,n_freq=args.n_freq, n=args.n).cuda().eval()
    E = Texture_Encoder(n_quad=args.nc_quad, nc_w=args.nc_w).cuda()

    G.load_state_dict(torch.load(os.path.join(dir,'models','G')),strict=False)
    E.load_state_dict(torch.load(os.path.join(dir,'models','E')),strict=False)
    z_to_w.load_state_dict(torch.load(os.path.join(dir,'models','z_to_w')),strict=False)

    G.eval()
    E.eval()
    z_to_w.eval()


    
    with torch.no_grad():

        
        config.args.batch_size=len(args_inference.text)+4
        loader,loader_test=get_loader()
        it=iter(loader)
        next(it)
        real=next(it).cuda()
        real_same=real[:1].repeat(config.args.batch_size,1,1,1)
        scale = Variable((.7+torch.randn(config.args.batch_size)*.0 ).cuda(),
                                 requires_grad=False)
        theta = Variable((0*torch.randn(config.args.batch_size)).cuda(),
                            requires_grad=False)
        real=augment(real,scale,theta)
        vgg(prep(real))
        out_vgg = [outputs[key] for key in layers] 
        w_real=E(out_vgg)

        
        print('different samplings')
        os.makedirs(os.path.join(dir_exp,'noise_sampling'))
        for i in range(8):
            rec=G(w_real[i:i+1].repeat(4,1))
            #grid=make_grid(rec*.5+.5,nrow=2,padding=5,pad_value=1)
            for j,img in enumerate(rec):
                save_image(img.unsqueeze(0)*.5+.5,os.path.join(dir_exp,'noise_sampling','noise_sampling_img%d_sample%d.png'%(i,j)))
            save_image(real[i:i+1]*.5+.5,os.path.join(dir_exp,'noise_sampling','noise_sampling_GT%d.png'%(i)))

        real_same=augment(real_same,scale,theta)
        vgg(prep(real_same))
        out_vgg = [outputs[key] for key in layers] 
        w_same=E(out_vgg)

        w_same+=torch.randn(w_same.shape).cuda()*.2

        
        #w_real=z_to_w(Variable(torch.randn(12,args.nc_z)).cuda())


        width=31
        kernel_size = [width,width] #kernel size is odd
        kernel = 1
        mgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        kernel=torch.sqrt((mgrids[0]-(kernel_size[0] - 1) / 2)**2+(mgrids[1]-(kernel_size[1] - 1) / 2)**2)
        kernel=torch.maximum(torch.tensor(0.),kernel_size[0]/2-kernel)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        conv = nn.Conv2d(1,1, kernel_size, groups=1, bias=False, stride=1, padding=int((kernel_size[0] - 1) / 2), padding_mode='zeros')
        conv.weight.data = kernel
        conv.weight.requires_grad = False
        conv.cuda()
        for w_curr,name in zip([w_real,w_same],['','_same_image']):
            print('brush%s %s'%(name,args_inference.text))
            w_map_back=w_curr[:4].T.unsqueeze(1).view(1,args.nc_w,2,2)
            w_map=F.interpolate(w_map_back, (128*(len(args_inference.text)//4),128*4), mode='bilinear', align_corners=True)
            for i,s in enumerate(args_inference.text.upper()):
                a=transforms.ToTensor()(Image.open('./letters/%s.png'%s)).unsqueeze(0).cuda()
                ca=conv(a)
                mask=torch.minimum(torch.tensor(1.).cuda(),ca/ca.max()*1.)
                save_image(mask,os.path.join(dir_exp,'mask.png'))
                w_brush=w_curr[4+i].unsqueeze(0)
                w_map[:,:,128*(i//4):128*((i//4)+1),128*(i%4):128*((i%4)+1)]=w_brush.unsqueeze(-1).unsqueeze(-1)*mask+w_map[:,:,128*(i//4):128*((i//4)+1),128*(i%4):128*((i%4)+1)]*(1-mask)

            brush=G(w_map[:,:,0,0],w_map=w_map,zoom=((len(args_inference.text)//4)*2,8))*.5+.5   #*2,8 for zommed
            save_image(brush,os.path.join(dir_exp,'brush%s.png'%(name)))
        
        

        config.args.batch_size=16
        loader,loader_test=get_loader()
        it=iter(loader)
        real=next(it).cuda()
        print('interpolation')
        scale = Variable((.7+torch.randn(config.args.batch_size)*.0 ).cuda(), requires_grad=False)
        theta = Variable((0*torch.randn(config.args.batch_size)).cuda(), requires_grad=False)
        vgg(prep(augment(next(it).cuda(),scale,theta)))
        out_vgg = [outputs[key] for key in layers] 
        w=E(out_vgg)
        for i in tqdm(range(4)):
            w_map=w[4*i:4*(i+1)].T.unsqueeze(0).view(1,args.nc_w,2,2)
            interp=G(w_map[:,:,0,0],w_map=w_map,zoom=(1,1))*.5+.5
            save_image(interp,os.path.join(dir_exp,'interpolation_%d.png'%i))

            interp=G(w_map[:,:,0,0],w_map=w_map,zoom=(2,2))*.5+.5
            save_image(interp,os.path.join(dir_exp,'interpolation_zoom_%d.png'%i))

        print('augmentation and reconstruction of the same image')
        next(it)
        batch=next(it).cuda()
        for i in tqdm(range(4)):
            real=batch[i:i+1].repeat(5,1,1,1)
            scale = Variable((torch.linspace(0,1,5) * (args.max_scale - args.min_scale) + args.min_scale).cuda(),
                                    requires_grad=False)
            theta = Variable((torch.linspace(0,0.5,5) * (args.max_theta - args.min_theta) + args.min_theta).cuda(),
                                requires_grad=False)
            real=augment(real,scale,theta)
            vgg(prep(real))  
            out_vgg = [outputs[key] for key in layers] 
            w=E(out_vgg)
            rec=G(w)
            cat=torch.cat((real,rec),dim=0)
            grid=make_grid(cat*.5+.5,nrow=5,padding=10,pad_value=1)
            save_image(grid,os.path.join(dir_exp,'augmentation_%d.png'%i))


    

        print('texture palette')

        batch=next(it).cuda()
        corner_list=[]
        w_real_list=[]
        
        for i in range(4):
            scale = Variable((torch.rand(1) * (args.max_scale - args.min_scale) + args.min_scale).cuda(),
                                requires_grad=False)
            theta = Variable((torch.rand(1) * (args.max_theta - args.min_theta) + args.min_theta).cuda(),
                                requires_grad=False)
            real=batch[i:i+1]
            aff = torch.stack(
            (torch.stack((torch.cos(theta) * scale, -torch.sin(theta) * scale, 0 * scale), dim=1),
            torch.stack((torch.sin(theta) * scale, torch.cos(theta) * scale, 0 * scale), dim=1)),
            dim=1)
            grid = F.affine_grid(aff, (1,3,700,700))
            t_real = F.grid_sample(real, grid)
            t_real = TF.center_crop(t_real, (int(torch.randint(350,512,(1,))),int(torch.randint(350,512,(1,)))))
            corner_list.append(t_real*.5+.5)
            vgg(prep(t_real))  
            out_vgg = [outputs[key] for key in layers] 
            w_real=E(out_vgg)
            w_real_list.append(w_real)

        w_real=torch.cat(w_real_list,dim=0)
        w=w_real[:4].T.unsqueeze(0).view(1,args.nc_w,2,2)
        w=F.pad(F.interpolate(w,scale_factor=4,mode='bilinear',align_corners=True),(2,2,2,2),mode='replicate')
        interp=G(w[:,:,0,0],w_map=w,zoom=(2,2))*.5+.5
        d=4
        w_map=F.interpolate(w,size=(interp.shape[-2],interp.shape[-1]), mode='bilinear', align_corners=True)
        boxes=[]
        w_interp=[]
        for (rx,ry) in [(.2,.5),(.5,.2),(.5,.8),(.8,.5)]:
            x,y=math.floor(rx*interp.shape[-2]),math.floor(ry*interp.shape[-1])
            w_interp.append(w_map[0,:,x,y])
            boxes.append([x-d,y-d,x+d,y+d])
        boxes=torch.tensor(boxes,dtype=torch.int)
        w_interp=torch.stack(w_interp,dim=0)
        img_interp=G(w_interp,zoom=(2,2))*.5+.5
        top=torch.cat((F.pad(corner_list[0].cpu(),(512-corner_list[0].shape[3],0,512-corner_list[0].shape[2],0),value=1),img_interp[0].unsqueeze(0).cpu(),F.pad(corner_list[1].cpu(),(0,512-corner_list[1].shape[3],512-corner_list[1].shape[2],0),value=1)),dim=3)
        middle=torch.cat((img_interp[1].unsqueeze(0).cpu(),draw_bounding_boxes((interp[0].cpu()*255).to(torch.uint8),boxes,width=5,colors=(255,0,0)).unsqueeze(0)/255,
        img_interp[2].unsqueeze(0).cpu()),dim=3)
        bottom=torch.cat((F.pad(corner_list[2].cpu(),(512-corner_list[2].shape[3],0,0,512-corner_list[2].shape[2]),value=1),img_interp[3].unsqueeze(0).cpu(),F.pad(corner_list[3].cpu(),(0,512-corner_list[3].shape[3],0,512-corner_list[3].shape[2]),value=1)),dim=3)
        save_image(torch.cat((top,middle,bottom),dim=2),os.path.join(dir_exp,'nuancier.png'))



            
        print('show off')
        real=next(it).cuda()
        scale = Variable((torch.rand(config.args.batch_size) * (args.max_scale - args.min_scale) + args.min_scale).cuda(),
                                requires_grad=False)
        theta = Variable((torch.rand(config.args.batch_size) * (args.max_theta - args.min_theta) + args.min_theta).cuda(),
                            requires_grad=False)
        real=augment(real,scale,theta)
        vgg(prep(real))  
        out_vgg = [outputs[key] for key in layers] 
        w=E(out_vgg)
        G.save_noise=False
        G(torch.zeros(1,args.nc_w).cuda()) #not the greatest way to do it, but clears the saved noise maps to prepare the zoomed gif
        G.save_noise=True
        w=E(out_vgg)

        w=w.view(4,4,args.nc_w)
        w=torch.cat((w,w[0].unsqueeze(0)),0)
        w=w.permute((2,1,0))

        w_path=F.interpolate(w,size=args_inference.n_gif+1,mode='linear',align_corners=True)
        w_path=w_path[...,:-1]
        w_path=w_path.permute((2,0,1)).view(args_inference.n_gif,args.nc_w,2,2)

        imgs_batches=[]
        #reduce args_inference.bs if memory problems here
        for b in tqdm(range(1+(args_inference.n_gif-1)//args_inference.bs)):
            imgs_batches.append(G(w_path[b*args_inference.bs:(b+1)*args_inference.bs,:,0,0],w_map=w_path[b*args_inference.bs:(b+1)*args_inference.bs],zoom=(2,2))*.5+.5)
        imgs=torch.cat(imgs_batches,dim=0)
        topil=transforms.ToPILImage()
        images = []
        for i in range(args_inference.n_gif):
            images.append(topil(imgs[i]))
        imageio.mimsave(os.path.join(dir_exp,'show_off.gif'), images,duration=.1)
        G.save_noise=False
        G(torch.zeros(1,args.nc_w).cuda()) #not the greatest way to do it, but clears the saved noise maps to prepare the zoomed gif






        print('examples')
        z = Variable(torch.randn(16,args.nc_z)).cuda()
        grid=make_grid(G(z_to_w(z)),nrow=4)*.5+.5
        save_image(grid,os.path.join(dir_exp,'examples.png'))

        print('zoom 4')
        z = Variable(torch.randn(1,args.nc_z)).cuda()
        w=z_to_w(z)
        zoom=G(w,zoom=(4,4))*.5+.5
        save_image(zoom,os.path.join(dir_exp,'zoom_4.png'))
        

        




       

        G.save_noise=True
        print('w_walk_same, between 8 transformations') #walk in w space between different augmentations of the same texture
        w_path=F.interpolate(w_same.T.unsqueeze(1),size=args_inference.n_gif+1,mode='linear',align_corners=True).squeeze(1).T
        w_path=w_path[:-1]
        imgs_batches=[]
        for b in tqdm(range(1+(args_inference.n_gif-1)//args_inference.bs)):
            imgs_batches.append(G(w_path[b*args_inference.bs:(b+1)*args_inference.bs])*.5+.5)
        imgs=torch.cat(imgs_batches,dim=0)
        topil=transforms.ToPILImage()
        images = []
        for i in range(args_inference.n_gif):
            images.append(topil(imgs[i]))
            
        imageio.mimsave(os.path.join(dir_exp,'w_walk_same.gif'), images,duration=.1)





        print('w_walk between 4 Encoding of real textures')
        real=next(it).cuda()
        scale = Variable((torch.rand(config.args.batch_size) * (args.max_scale - args.min_scale) + args.min_scale).cuda(),
                                requires_grad=False)
        theta = Variable((torch.rand(config.args.batch_size) * (args.max_theta - args.min_theta) + args.min_theta).cuda(),
                            requires_grad=False)
        real=augment(real,scale,theta)
        t_real=augment(real,scale,theta)
        vgg(prep(t_real))  
        out_vgg = [outputs[key] for key in layers] 
        w_real=E(out_vgg)
        w=w_real[:4]
        w=torch.cat((w,w[0].unsqueeze(0)),0)

        w=w.T.unsqueeze(1)
        w_path=F.interpolate(w,size=args_inference.n_gif+1,mode='linear',align_corners=True).squeeze(1).T
        w_path=w_path[:-1]
        imgs_batches=[]
        for b in tqdm(range(1+(args_inference.n_gif-1)//args_inference.bs)):
            imgs_batches.append(G(w_path[b*args_inference.bs:(b+1)*args_inference.bs])*.5+.5)
        imgs=torch.cat(imgs_batches,dim=0)
        topil=transforms.ToPILImage()
        images = []
        for i in range(args_inference.n_gif):
            images.append(topil(imgs[i]))
        imageio.mimsave(os.path.join(dir_exp,'w_walk.gif'), images,duration=.1)

        print('z_walk between 4 random noise samples in Z space')
        z = z.T.unsqueeze(1)
        z_path=F.interpolate(z,size=args_inference.n_gif+1,mode='linear',align_corners=True)
        z_path=z_path[...,:-1]
        z_path=z_path.squeeze(1).T
        imgs_batches=[]
        for b in tqdm(range(1+(args_inference.n_gif-1)//args_inference.bs)):
            imgs_batches.append(G(z_to_w(z_path[b*args_inference.bs:(b+1)*args_inference.bs]))*.5+.5)
        imgs=torch.cat(imgs_batches,dim=0)
        images = []
        for i in range(args_inference.n_gif):
            images.append(topil(imgs[i]))
        imageio.mimsave(os.path.join(dir_exp,'z_walk.gif'), images,duration=.1)


        
        G.save_noise=False

        print('reconstruction samples')
        it=iter(loader)
        l=[]
        for _ in tqdm(range(8)):
            real=next(it).cuda()
            scale = Variable((torch.rand(config.args.batch_size) * (args.max_scale - args.min_scale) + args.min_scale).cuda(),
                                requires_grad=False)
            theta = Variable((torch.rand(config.args.batch_size) * (args.max_theta - args.min_theta) + args.min_theta).cuda(),
                                requires_grad=False)     
            t_real=augment(real,scale,theta)  
            vgg(prep(t_real))  
            out_vgg = [outputs[key] for key in layers] 
            w=E(out_vgg)
            rec=G(w)
            l.append(torch.cat((t_real,rec),dim=3))
        cat=torch.cat(l,dim=0)
        grid=make_grid(cat,nrow=8)*.5+.5
        save_image(grid.cpu(),os.path.join(dir_exp,'reconstruction.png'))








