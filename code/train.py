import matplotlib
matplotlib.use('Agg')
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import os
import torch.fft as fft
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from util import *
from models import *
import config

if __name__ == '__main__':
    

    args=config.get_arguments()
    config.create_dir(args)
    config.save_args()
    torch.manual_seed(config.args.seed)
    print(config.args)

    gblur = GaussianSmoothing(3, 5, 1.7).cuda()
    writer = SummaryWriter(args.dir)
    
    # for style loss:
    vgg = models.vgg19(pretrained=False).features.cuda()
    try:
        pretrained_dict = torch.load('./vgg.pth')
        for param, item in zip(vgg.parameters(), pretrained_dict.keys()):
            param.data = pretrained_dict[item].type(torch.FloatTensor).cuda()
    except:
        pretrained_dict = torch.load('../vgg.pth')
        for param, item in zip(vgg.parameters(), pretrained_dict.keys()):
            param.data = pretrained_dict[item].type(torch.FloatTensor).cuda()

    #vgg = models.vgg19(pretrained=True).features.cuda()

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


    # Initialize networks
    E = Texture_Encoder(n_quad=args.nc_quad, nc_w=args.nc_w).cuda()
    z_to_w=Z_to_w(nc_z=args.nc_z, nc_w=args.nc_w,depth=args.depth_T).cuda()
    G = style_generator(nc_w=args.nc_w,n_freq=args.n_freq,n=args.n).cuda()
    WD=W_Discriminator(args.nc_w).cuda()
    classes=torch.cat((torch.ones(args.batch_size,1),-torch.ones(args.batch_size,1)),dim=0).cuda() #for the discriminator

    if config.args.load is not None:
        try:
            G.load_state_dict(torch.load(os.path.join(config.args.dir,'models','G')),strict=False)
            E.load_state_dict(torch.load(os.path.join(config.args.dir,'models','E')),strict=False)
            z_to_w.load_state_dict(torch.load(os.path.join(config.args.dir,'models','z_to_w')),strict=False)
            
        except:
            pass
    n_it=config.args.it




    # Optimizers
    optimizer = optim.Adam(list(G.parameters()) + list(E.parameters()) +list(z_to_w.parameters()), lr=args.lr, betas=(args.beta1, 0.999),weight_decay=0)
    optimizer_WD = optim.Adam(WD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Constant variables for consistent ploting
    loader,loader_test = get_loader(shuffle=False)
    
    z_plot = Variable(torch.rand(args.batch_size, args.nc_z).cuda())
    scale_plot = Variable((torch.linspace(0, 1, 2 * args.batch_size) * (args.max_scale - args.min_scale) + args.min_scale).cuda(),requires_grad=False)
    theta_plot = Variable((torch.linspace(0, 1, 2 * args.batch_size) * (args.max_theta - args.min_theta) + args.min_theta).cuda(),requires_grad=False)
    it = iter(loader)
    real_plot=Variable(torch.cat((next(it), next(it)), dim=0).cuda(), requires_grad=False)
    t_real_plot=augment(real_plot,scale_plot,theta_plot)
    it = iter(loader_test)
    test_plot=Variable(torch.cat((next(it), next(it)), dim=0).cuda(), requires_grad=False)
    t_test_plot=augment(test_plot,scale_plot,theta_plot)
    with torch.no_grad():
        vgg(prep(t_real_plot))
        out_vgg_plot = [outputs[key] for key in layers] 
        vgg(prep(t_test_plot))
        out_vgg_test_plot = [outputs[key] for key in layers] 

    loader,loader_test = get_loader(shuffle=True)


    print('G has %d parameters, E has %d parameters, T has %d parameters'%(get_n_params(G),get_n_params(E),get_n_params(z_to_w)))
    print('start training...')
    hist_scale, hist_theta = [], []
    
    for epoch in range(args.n_epoch):
        for step, real in enumerate(loader):
            real = Variable(real.cuda(), requires_grad=False)
            n_it += 1
            E.zero_grad()
            G.zero_grad()
            with torch.no_grad():
                scale = Variable((torch.rand(args.batch_size) * (args.max_scale - args.min_scale) + args.min_scale).cuda(),requires_grad=False)
                theta = Variable((torch.rand(args.batch_size) * (args.max_theta - args.min_theta) + args.min_theta).cuda(),requires_grad=False)
                t_real=augment(real,scale,theta) # t_real is augmented with random rotation and scaling.

                real_same = real[:1].repeat(real.shape[0], 1, 1, 1)
                t_real_same=augment(real_same,scale,theta) #t_real is the SAME image augmented with random rotation and scaling.

                vgg(prep(t_real))
                out_vgg_real = [outputs[key] for key in layers] 
                style_targets = [GramMatrix()(outputs[key]) for key in layers] 

                vgg(prep(t_real_same))
                out_vgg_real_same = [outputs[key] for key in layers]

            w_same = E(out_vgg_real_same) # latent codes for different augmentations of the same image

            # self supervised learning inspired loss: consistency between geometric tranformations and predicted transformations
            s, t = G.pred(w_same) # parameters prediction
            loss_scale = torch.mean((s / s[:1] - (scale / scale[:1])) ** 2)
            loss_theta = torch.mean(anglel1(t - t[:1], (theta - theta[:1]))[1:])
            loss_pred = loss_scale + loss_theta 
            writer.add_scalar('loss_scale', loss_scale.item(), n_it)
            writer.add_scalar('loss_theta', loss_scale.item(), n_it)
            (args.lam_pred * loss_pred).backward() 
            


            optimizer.step() #updates E and G.pred
            E.zero_grad()
            G.zero_grad()
            z_to_w.zero_grad()

            w = E(out_vgg_real)
            
            if args.lam_w !=0: #trains z_to_w, but not E
                z=Variable(torch.randn(args.batch_size,args.nc_z).cuda(),requires_grad=False)
                w_fake=z_to_w(z)
                batch_w=torch.cat((w.detach(),w_fake),dim=0)
                loss_latent = -(WD(batch_w)*classes).mean()
                (args.lam_w*loss_latent).backward(retain_graph=True)
            

            # Measure the spread of latent codes for plotting, default does not minimize this value
            reg=torch.mean(w**2)
            (args.lam_reg*reg).backward(retain_graph=True)
            writer.add_scalar('reg', reg.item(), n_it)


            # Color histogram matching loss
            rec = G(w)
            hist_loss=Hist_loss(I(rec),t_real)
            (args.lam_hist * hist_loss).backward(retain_graph=True)
            writer.add_scalar('hist_loss', hist_loss.item(), n_it)


            hist_scale, hist_theta = hist_scale + list(G.scale.detach().cpu()), hist_theta + list(G.theta.detach().cpu())  

            l1_loss = sum([m.l1_loss for m in G.body_modules])   # l1_loss is by default not minimized. Its objective is to minimize the use of the sine maps in the network,
            (args.lam_l1 * l1_loss).backward(retain_graph=True)  # it was implemented because some periodicity artifacts were found in stochastic textures
            writer.add_scalar('l1_loss', l1_loss.item(), n_it)
            if args.texture_loss == 'gatys':  # gatys
                if args.gradnorm: # normalizes the L1 norm of the gradient of the image with respect to the style loss for each of the 5 layers chosen by Gatys et. al.
                                  # durinig training all 5 'scales' will have the same impact on the image morphologically
                    vgg(prep(I(rec)))  
                    out_vgg_rec = [outputs[key] for key in layers] 
                    style_losses = [layers_weights[a] * GramMSELoss()(A, style_targets[a].detach()) / len(out_vgg_rec) for a, A
                                    in enumerate(out_vgg_rec)]
                    for i, style_loss in enumerate(style_losses):
                        (style_loss).backward(retain_graph=True)
                else:
                    vgg(prep(rec))
                    out_vgg_rec = [outputs[key] for key in layers] 
                    style_losses = [layers_weights[a] * GramMSELoss()(A, style_targets[a].detach()) / len(out_vgg_rec) for a, A
                                    in enumerate(out_vgg_rec)]
                    (args.lam_style*sum(style_losses)).backward(retain_graph=True)

                writer.add_scalar('style_loss', sum(style_losses).item(), n_it)

                if True: # Spectral loss 
                    lum=torch.randn(3,1).cuda()
                    lum= lum / lum.norm(2)
                    grey = rec.permute(0, 2, 3, 1).matmul(lum).permute(0, 3, 1, 2) # project to grey using a random direction in color space
                    f = fft.fft2(grey)
                    f_proj = f / (f.abs() + 1e-8) * fft.fft2(
                        t_real.permute(0, 2, 3, 1).matmul(lum).permute(0, 3, 1, 2)).abs()
                    proj = fft.ifft2(f_proj).real.detach() # create an grey image with the phase from rec and the module from the original image
                    loss_sp = ((grey - proj) ** 2).mean()  # spectral loss in image space
                    writer.add_scalar('loss_spe/', loss_sp.item(), n_it)
                    (args.lam_sp*loss_sp).backward(retain_graph=True)

                (0. * w.mean()).backward() # not pretty, but clears the graph after backward()
                optimizer.step()

            elif args.texture_loss == 'snelgrove':
                
                for i, (gt_blur, rec_blur) in enumerate(zip(gblur(t_real, 4), gblur(rec, 4))):
                    with torch.no_grad():
                        vgg(prep(gt_blur))
                        out_vgg_real = [outputs[key] for key in [1,6,11]] 
                        style_targets = [GramMatrix()(outputs[key]) for key in [1,6,11]]
                    if args.gradnorm:    
                        vgg(prep(I(rec)))  
                        out_vgg_rec = [outputs[key] for key in [1, 6, 11]] 
                        style_losses = [[1/n**2 for n in [64,128,256]][a] * GramMSELoss()(A, style_targets[a].detach()) / len(out_vgg_rec) for a, A
                                        in enumerate(out_vgg_rec)]
                        for _, style_loss in enumerate(style_losses):
                            (style_loss).backward(retain_graph=True)
                    else:
                        vgg(prep(rec))  
                        out_vgg_rec = [outputs[key] for key in [1, 6, 11]] 
                        style_losses = [[1/n**2 for n in [64,128,256]][a] * GramMSELoss()(A, style_targets[a].detach()) / len(out_vgg_rec) for a, A
                                        in enumerate(out_vgg_rec)]
                        (args.lam_style*sum(style_loss)).backward(retain_graph=True)
                    writer.add_scalar('style_loss/scale_%d' % i, sum(style_losses).item(), n_it)
                if args.lam_sp != 0:
                    lum=torch.randn(3,1).cuda()
                    lum= lum / lum.norm(2)
                    grey = rec.permute(0, 2, 3, 1).matmul(lum).permute(0, 3, 1, 2) # project to grey using a random direction in color space
                    f = fft.fft2(grey)
                    f_proj = f / (f.abs() + 1e-8) * fft.fft2(
                        t_real.permute(0, 2, 3, 1).matmul(lum).permute(0, 3, 1, 2)).abs()
                    proj = fft.ifft2(f_proj).real.detach() # create an grey image with the phase from rec and the module from the original image
                    loss_sp = ((grey - proj) ** 2).mean()  # spectral loss in image space
                    writer.add_scalar('error/', loss_sp.item(), n_it)
                    (args.lam_sp*loss_sp).backward(retain_graph=True)

                (0. * w.mean()).backward() # not pretty, but clears the graph after backward()
                optimizer.step()





            if args.lam_w !=0 : #train WD
                for _ in range(3):
                    WD.zero_grad()
                    w_fake=z_to_w(z)
                    batch_w=torch.cat((w,w_fake),dim=0)
                    loss_latent = (WD(batch_w.detach())*classes).mean()
                    gp_latent = calc_gradient_penalty(WD, w.detach(),w_fake.detach())
                    (args.lam_w*(loss_latent+gp_latent)).backward()
                    optimizer_WD.step()
                writer.add_scalar('loss_WD', loss_latent.item(), n_it)


        #save logs
        if (epoch % args.print_every) == 0:
            G.eval()
            E.eval()
            config.args.it=n_it
            config.save_args()
            torch.save(G.state_dict(), os.path.join(args.dir,'models','G'))
            torch.save(E.state_dict(), os.path.join(args.dir,'models','E'))
            torch.save(z_to_w.state_dict(), os.path.join(args.dir,'models','z_to_w'))

            with torch.no_grad(): #logging
                w = E(out_vgg_plot)
                z = Variable(torch.rand(args.batch_size * 2, args.nc_z).cuda())
                rec = G(w)

            
                lum = torch.tensor([[0.29900],[0.58700],[0.11400]]).cuda()
                lum = lum / lum.norm(2)
                grey = rec.permute(0, 2, 3, 1).matmul(lum).permute(0, 3, 1, 2)
                f = fft.fft2(grey)
                f_proj = f / (f.abs() + 1e-8) * fft.fft2(
                    t_real_plot.permute(0, 2, 3, 1).matmul(lum).permute(0, 3, 1, 2)).abs()
                proj = fft.ifft2(f_proj).real.detach()
                grid = torch.cat((make_grid(F.interpolate(grey, scale_factor=2, mode='nearest'), nrow=1),
                                  make_grid(F.interpolate(proj, scale_factor=2, mode='nearest'), nrow=1)), dim=2)
                writer.add_image('proj', torch.tanh(grid) * .5 + .5, n_it) 
                

                cat=torch.cat((TF.center_crop(t_real_plot, 256),rec),dim=3)
                cat=F.interpolate(cat, scale_factor=2, mode='nearest')
                grid=make_grid(cat,nrow=1)
                writer.add_image('Texture reconstruction', grid * .5 + .5, n_it)
                save_image(grid*.5+.5,os.path.join(args.dir,'inference','reconstruction.png'))


                cat=torch.cat((TF.center_crop(t_test_plot, 256),G(E(out_vgg_test_plot))),dim=3)
                cat=F.interpolate(cat, scale_factor=2, mode='nearest')
                grid=make_grid(cat,nrow=1)
                writer.add_image('Test reconstruction', grid * .5 + .5, n_it)


                grid=make_grid(G(z_to_w(z_plot)),nrow=4)
                writer.add_image('Direct_sampling', grid * .5 + .5, n_it)


                writer.add_histogram('predicted scales', torch.log2(torch.tensor(hist_scale)), n_it)
                writer.add_histogram('predicted angles', torch.tensor(hist_theta), n_it)
                hist_scale, hist_theta = [], []
               
              
                fig, ax = plt.subplots(dpi=300)
                mod, phase = 2 ** (-G.r * G.grad_boost), G.phase * G.grad_boost
                fx, fy = mod * torch.cos(phase), mod * torch.sin(phase) 
                ax.plot(fx.detach().cpu(), fy.detach().cpu(), "x", markersize=.5, color=(1, 0, 0))
                circle = plt.Circle((0, 0), 0.5, color='r', fill=False, linewidth=.2)
                ax.add_artist(circle)
                for i, m in enumerate(G.body_modules.children()):
                    circle = plt.Circle((0, 0), 2 ** (-i - 3), color='r', fill=False, linewidth=.2)
                    ax.add_artist(circle)
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.axis('off')
                ax.set_aspect('equal')
                writer.add_figure('frequencies', fig, n_it)


                plt.close('all')
                writer.flush()
            G.train()
            E.train()
                

    torch.save(G.state_dict(), os.path.join(args.dir,'models','G'))
    torch.save(E.state_dict(), os.path.join(args.dir,'models','E'))
    torch.save(z_to_w.state_dict(), os.path.join(args.dir,'models','z_to_w'))
    config.args.it=n_it
    config.save_args()
    os.system('python inference.py --name %s' %(args.name))
    writer.flush()
    writer.close()