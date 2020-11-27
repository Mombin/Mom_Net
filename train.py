import matplotlib
matplotlib.use('Agg')
import argparse
import os
from pathlib import Path
import random
import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.utils as vutils
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
import numpy as np
import pytorch_msssim


import matplotlib.pyplot as plt

from networks import autoencoder as AE
from networks import DisneyNet as DSN



n_label = 1
code_size = 512 - n_label
batch_size = 16
n_critic = 1

parser = argparse.ArgumentParser()

parser.add_argument('--bg_data','--bg_dir', type=str, dest='dst_path', default='/home/jjck5938/test_workspace/data_dst')
parser.add_argument('--tf_data','--tf_dir', type=str, dest='src_path', default='/home/jjck5938/test_workspace/data_src')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batch', type=int, dest='batch_size', default=64)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--num_channel', type=int, dest='nc', default=3)
parser.add_argument('--num_z', type=int, dest='nz', default=100)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--nef', type=int, default=128)
parser.add_argument('--epoch',type=int, dest='num_epochs',default=100)
parser.add_argument('--lr', type=float, default=1e-3) 
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--result', type=str, dest='result_path', default='/home/jjck5938/Mom_Net/results')
parser.add_argument('--save_path', dest="save_model",type=str, default='aemom_sim.pth')
parser.add_argument('--checkpoint', type=str, default='/home/jjck5938/Mom_Net/checkpoints')
args = parser.parse_args()



#load dataset
datasetA = dset.ImageFolder(root=args.dst_path,
                            transform = transforms.Compose([
                                transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

dataloaderA = torch.utils.data.DataLoader(datasetA, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

datasetB = dset.ImageFolder(root=args.src_path,
                            transform=transforms.Compose([
                                transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

def dfl_loader(path):
    def loader(transform):
        data = dset.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=args.batch_size,
                                    num_workers=args.workers)
        return data_loader   
    return loader



def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)
    
    for img, label in loader:
        yield img, label




#device = torch.device("cuda:1" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
#device2 = torch.device("cuda:1" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")


"""
#Plot some training image
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=1, normalize=True).cpu(), (1,2,0)))
plt.savefig(os.path.join(args.result_path, "real_sample.png"))
"""

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def copyParams(module_src, module_dest):
    params_src = module_src.named_parameters()
    params_dest = module_dest.named_parameters()

    dict_dest = dict(params_dest)

    for name, param in params_src:
        if name in dict_dest:
            dict_dest[name].data.copy_(param.data)

#step = 0
#total_step = 8
#wsize = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netESRC = DSN.Encoder_(args, n_label)#.to(device1) 
netEDST = DSN.Encoder_(args, n_label)#.to(device2)

netDSRC = DSN.Decoder_(args, code_size, n_label)#.to(device1)
netDDST = DSN.Decoder_(args, code_size, n_label)#.to(device2)


model_src = DSN.autoencoder(args, netESRC, netDSRC)#.to(device1)#.cuda()
model_dst = DSN.autoencoder(args, netEDST, netDDST)#.to(device2)#.cuda()


resume = Path(args.save_model)


if torch.cuda.device_count() > 1:
    netESRC = nn.DataParallel(netESRC)
    netEDST = nn.DataParallel(netEDST)
    netDSRC = nn.DataParallel(netDSRC)
    netDDST = nn.DataParallel(netDDST)
    model_src = nn.DataParallel(model_src)
    model_dst = nn.DataParallel(model_dst)

netESRC.to(device)
netEDST.to(device)
netDSRC.to(device)
netDDST.to(device)
model_src.to(device)
model_dst.to(device)

criterion = nn.MSELoss()

optimizerSRC = torch.optim.Adam(
    model_src.parameters(), lr=args.lr, weight_decay=1e-5)
optimizerDST = torch.optim.Adam(
    model_dst.parameters(), lr=args.lr, weight_decay=1e-5)


def train(ae_src, ae_dst, en_src, en_dst, loader_src, loader_dst, p_list):
    
    step = p_list['step']
    total_step = 8
    wsize = 3
    
    dset_src = sample_data(loader_src, 4 * 2 ** step)
    dset_dst = sample_data(loader_dst, 4 * 2 ** step)
    pbar = tqdm(range(1000000))
    alpha = p_list['alpha']
    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    iteration = p_list['iteration']
    stablize = False

    for i in pbar:
        start_time=time.time()
        ae_src.zero_grad()

        aplha = min(1, 0.00002 * iteration)

        if stablize is False and iteration > 50000:
            dset_src = sample_data(loader_src, 4 * 2 ** step)
            dset_dst = sample_data(loader_dst, 4 * 2 ** step)
            stablize = True

        if iteration > 100000:
            alpha = 0
            iteration = 0
            step += 1
            stabilize = False

            if step > 8:
                alpha = 1
                step = 8
            dset_src = sample_data(loader_src, 4 * 2 ** step)
            dset_dst = sample_data(loader_dst, 4 * 2 ** step)

        if step > 2:
            wsize += 4
            if wsize > 11:
                wsize = 11

        try:
            src_image, src_label = next(dset_src)
            dst_image, dst_label = next(dset_dst)
            #print(dst_image)
        
        except (OSError, StopIteration):
            dset_src = sample_data(loader_src, 4 * 2 ** step)
            dset_dst = sample_data(loader_dst, 4 * 2 ** step)
            src_image, src_label = next(dset_src)
            dst_image, src_label = next(dset_dst)


        
        iteration += 1

        #b_size = src_image.size(0)
        #src_image = Variable(src_image).cuda()
        #src_label = Variable(src_label).to(device)
        #dst_image = Variable(dst_image).cuda()
        #dst_label = Variable(dst_label).to(device)

        src_image = src_image.clone().detach().requires_grad_(True).cuda()
        dst_image = dst_image.clone().detach().requires_grad_(True).cuda()
        src_predict = ae_src(src_image, step, alpha)
        dst_predict = ae_dst(dst_image, step, alpha)
        dst_in_src = ae_src(dst_image, step, alpha)
        src_in_dst = ae_dst(src_image, step, alpha)
        
        #src_predict = src_predict.mean() - 0.001 * (src_predict ** 2).mean()
        #dst_predict = src_predict.mean() - 0.001 * (dst_predict ** 2).mean()
        #dst_predict.backward(mone)
        #src_predict.backward(mone)
        #eps = torch.rand(b_size, 1, 1, 1).cuda()
        

        #==================backward==================
        ssim_loss = pytorch_msssim.SSIM(window_size=wsize)
        msssim_loss = pytorch_msssim.MSSSIM()

        #if step <= 2:

        loss_src = 1-ssim_loss(src_image, src_predict)
        loss_dst = 1-ssim_loss(dst_image, dst_predict)

        #elif step> 2:
        #    loss_src = -msssim_loss(src_image, src_predict)
        #    loss_dst = -msssim_loss(dst_image, dst_predict)


        optimizerSRC.zero_grad()
        loss_src.backward()
        optimizerSRC.step()

        with torch.no_grad():
            copyParams(ae_src.module.enc, ae_dst.module.enc)
            #copyParams(en_src, en_dst)

        optimizerDST.zero_grad()
        loss_dst.backward()
        optimizerDST.step()

        
        with torch.no_grad():
            copyParams(ae_dst.module.enc, ae_src.module.enc)
            #copyParams(en_dst, en_src)


        pbar.set_description(
            (f'{i}, iter:{iteration + 1}, SRC: {-loss_src:.5f}, DST: {-loss_dst:.5f},'
             f' Step:{step:d}/{total_step:d}'))

        if i %1000 ==0:
            pic1 = dst_predict.cpu().data
            pic2 = src_predict.cpu().data
            pic_dsrc = src_in_dst.cpu().data
            pic_sdst = dst_in_src.cpu().data

            pic_out = torch.cat([pic1, pic2, pic_dsrc, pic_sdst], dim=0)
            #save_image(pic1, f'./mlp_img/result_dst_{i}iter_{step}step.png')
            #save_image(pic2, f'./mlp_img/result_src_{i}iter_{step}step.png')
            save_image(pic_out, f'./mlp_img/result{i}iter_{step}step.png')
            torch.save({
                        'netESRC_state_dict' : netESRC.state_dict(),
                        'netEDST_state_dict' : netEDST.state_dict(),
                        'netDSRC_state_dict' : netDSRC.state_dict(),
                        'netDDST_state_dict' : netDDST.state_dict(),
                        'modelSRC_state_dict': ae_src.state_dict(),
                        'modelDST_state_dict': ae_dst.state_dict(),
                        'optimizerSRC_state_dict': optimizerSRC.state_dict(),
                        'optimizerDST_state_dict': optimizerDST.state_dict(),
                        'step': step,
                        'iteration': iteration,
                        'alpha': alpha
                        }, args.save_model)
            print("save success!")
            if i+1 % 10000 == 0:
                save_name = f"cp{iteration}iter_{step}step.pth"
                torch.save({
                        'netESRC_state_dict' : netESRC.state_dict(),
                        'netEDST_state_dict' : netEDST.state_dict(),
                        'netDSRC_state_dict' : netDSRC.state_dict(),
                        'netDDST_state_dict' : netDDST.state_dict(),
                        'modelSRC_state_dict': ae_src.state_dict(),
                        'modelDST_state_dict': ae_dst.state_dict(),
                        'optimizerSRC_state_dict': optimizerSRC.state_dict(),
                        'optimizerDST_state_dict': optimizerDST.state_dict(),
                        'step': step,
                        'iteration': iteration,
                        'alpha': alpha
                        }, os.path.join(args.checkpoint, save_name))


if __name__ == '__main__':
    src_loader = dfl_loader(args.src_path)
    dst_loader = dfl_loader(args.dst_path)

    params = {'step':0, 'total_ste':8, 'w_size':3, 'iteration':0, 'alpha':0} 

    if Path.exists(resume):
        isresume = input(str(resume) +"already exists do you want to continue training(y)? or init(n)?")
        assert isresume == 'y' or isresume == 'n' , 'you give wrong input please restart code!!'
        if isresume == 'y':
            cpt = torch.load(resume)
            netESRC.load_state_dict(cpt['netESRC_state_dict'])
            netEDST.load_state_dict(cpt['netEDST_state_dict'])
            netDSRC.load_state_dict(cpt['netDSRC_state_dict'])
            netDSRC.load_state_dict(cpt['netDSRC_state_dict'])
            model_src.load_state_dict(cpt['modelSRC_state_dict'])
            model_dst.load_state_dict(cpt['modelDST_state_dict'])
            optimizerSRC.load_state_dict(cpt['optimizerSRC_state_dict'])
            optimizerDST.load_state_dict(cpt['optimizerDST_state_dict'])
            params['step'] = cpt['step']
            params['iteration'] = cpt['iteration']
            params['alpha'] = cpt['alpha']
            
        elif isresume == 'n':
            print("initialize training")
    else:
        print("start training first")


    train(ae_src=model_src, ae_dst=model_dst, en_src=netESRC, en_dst= netEDST, loader_src=src_loader, loader_dst=dst_loader, p_list=params)
