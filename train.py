import matplotlib
matplotlib.use('Agg')
import argparse
import os
import random
import time

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

import matplotlib.pyplot as plt

from models import autoencoder as AE



#Set random seed for reproducibility
manualSeed = 999

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


parser = argparse.ArgumentParser()

parser.add_argument('--bg_data','--bg_dir', type=str, dest='data_dst', default='data/data_dst')
parser.add_argument('--tf_data','--tf_dir', type=str, dest='data_src', default='data/data_src')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batch', type=int, dest='batch_size', default=64)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--num_channel', type=int, dest='nc', default=3)
parser.add_argument('--num_z', type=int, dest='nz', default=100)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--nef', type=int, default=128)
parser.add_argument('--epoch',type=int, dest='num_epochs',default=100)
parser.add_argument('--lr', type=float, default=1e-3) 
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--result', type=str, dest='result_path', default='/home/jjck5938/changed_model/results')
parser.add_argument('--checkpoint', dest="save_model",type=str, default='aemom_sim.pth')
args = parser.parse_args()


datasetA = dset.ImageFolder(root=args.data_dst,
                            transform=transforms.Compose([
                                transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

dataloaderA = torch.utils.data.DataLoader(datasetA, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

datasetB = dset.ImageFolder(root=args.data_src,
                            transform=transforms.Compose([
                                transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)




device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")


"""
#Plot some training image
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=1, normalize=True).cpu(), (1,2,0)))
plt.savefig(os.path.join(args.result_path, "real_sample.png"))
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def to_img(x):
    x = 0.5 * (x +1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


netE = AE.encoder_(args).cuda() 
netDA = AE.decoder_(args).cuda()
netDB = AE.decoder_(args).cuda()

modelA = AE.autoencoder(args, netE, netDA).cuda()
modelB = AE.autoencoder(args, netE, netDB).cuda()

criterion = nn.MSELoss()

optimizerA = torch.optim.Adam(
    modelA.parameters(), lr=args.lr, weight_decay=1e-5)
optimizerB = torch.optim.Adam(
    modelB.parameters(), lr=args.lr, weight_decay=1e-5)



for epoch in range(args.num_epochs):
    start_time=time.time()
    for i, (data, _) in enumerate(dataloaderA):
        img = data
        img = Variable(img).cuda()
        #=================forward==================
        #outputE = netE(img)
        #outputDA = netDA(outputE)
        outputA = modelA(img)
        lossA = criterion(outputA, img)
        #=================backward=================
        optimizerA.zero_grad()
        lossA.backward()
        optimizerA.step()

    for i, (data, _) in enumerate(dataloaderB):
        #print("{}steps".format(i))
        img = data
        img = Variable(img).cuda()
        # ================forward==================
        outputB = modelB(img)
        lossB = criterion(outputB, img)
        #=================backward=================
        optimizerB.zero_grad()
        lossB.backward()
        optimizerB.step()

    print('epoch [{}/{}], lossA:{:.4f}, lossB:{:.4f}, time:{}s'
            .format(epoch+1, args.num_epochs, lossA.item(),lossB.item(), int(time.time()-start_time)))

    if epoch %10 ==0:
        pic = outputA.cpu().data
        save_image(pic, './mlp_img/image_outA{}.png'.format(epoch))  
        print("save success!")
        pic2 = outputB.cpu().data
        save_image(pic2, './mlp_img/image_outB{}.png'.format(epoch))
        torch.save({
                    'modelA_state_dict': modelA.state_dict(),
                    'modelB_state_dict': modelB.state_dict(),
                    'netE_state_dict' : netE.state_dict(),
                    'netDA_state_dict' : netDA.state_dict(),
                    'netDB_state_dict' : netDB.state_dict(),
                    'optimizerA_state_dict': optimizerA.state_dict(),
                    'optimizerB_state_dict': optimizerB.state_dict()
                    }, args.save_model)
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'netE_state_dict' : netE.state_dict(),
            'netDA_state_dict' : netDA.state_dict(),
            'netDB_state_dict' : netDB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict()
            }, args.save_model)
