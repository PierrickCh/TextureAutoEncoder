import os
import json
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_runs', help='directory with all experiments', default='./runs')
    parser.add_argument('--dirname', help='experiment directory', default='test')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--center_gram', type=str2bool,default=True)
    parser.add_argument('--gradnorm', type=str2bool,default=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=10 ** -4)
    parser.add_argument('--beta1', type=float, default=.9)
    parser.add_argument('--local_stats_width', type=float, default=.2)

    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--nc_w', type=int, default=32)
    parser.add_argument('--nc_z', type=int, default=256)
    parser.add_argument('--nc_quad', type=int, default=32)
    parser.add_argument('--nc_max', type=int, default=128)
    parser.add_argument('--depth_T', type=int, default=5)

    parser.add_argument('--lam_style', type=float, default=1)
    parser.add_argument('--lam_sp', type=float, default=1.)
    parser.add_argument('--lam_hist', type=float, default=1.)
    parser.add_argument('--lam_pred', type=float, default=1000.)
    parser.add_argument('--lam_w', type=float, default=100)

    parser.add_argument('--lam_l1', type=float, default=0.)
    
    parser.add_argument('--lam_reg', type=float, default=0)
    
    

    parser.add_argument('--min_scale', type=float, default=.3)
    parser.add_argument('--max_scale', type=float, default=1)
    parser.add_argument('--min_theta', type=float, default=0)
    parser.add_argument('--max_theta', type=float, default=6.28)

    parser.add_argument('--dataset_folder', type=str, default='../../../data/MacroTextures500/train/',help='path to all your data, train and val are done randomly')
    parser.add_argument('--texture_loss', type=str, default='gatys')
    parser.add_argument('--print_every', type=int, default=100)
    
    parser.add_argument('--n', type=int, default=7,help='depth of the generator')
    parser.add_argument('--n_freq', type=int, default=32)
    parser.add_argument('--sine_maps', type=str2bool,default=True)
    parser.add_argument('--freq_amp', type=str, default='2scales')
    parser.add_argument('--sine_maps_merge', type=str, default='add')
    

    parser.add_argument('--load', type=str, default=None)

    args, unknown = parser.parse_known_args()
    if args.load is not None:
        args=load_args(args.load,args)
    
    return args

def create_dir():
    dir=os.path.join(args.dir_runs,args.dirname)
    if not os.path.exists(dir):
        os.makedirs(dir)
        args.dir=dir
    else:
        dir_search=dir
        i=0
        while os.path.exists(dir_search):
            i+=1
            dir_search=dir+'_%d'%i
        os.makedirs(dir_search)
        args.dir=dir_search
    os.makedirs(os.path.join(args.dir,'models'))
    os.makedirs(os.path.join(args.dir,'inference'))
    return

def save_args():
    with open(os.path.join(args.dir,'arguments.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return

def load_args(path,args):
    #args, unknown = argparse.ArgumentParser().parse_known_args()
    with open(path, 'r') as f:
        args.__dict__.update(json.load(f))
    return args
