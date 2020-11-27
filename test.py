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

from networks import autoencoder as AE


#Set random seed for reproducibility
manualSeed = 999

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


parser = argparse.ArgumentParser()

parser.add_argument('--bg_data', type=str, dest='data_dst', default='data/data_dst')
parser.add_argument('--target_data', type=str, dest='data_src', default='data/data_src')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batch', type=int, dest='batch_size', default=128)
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
parser.add_argument('--checkpoint', type=str, default='aemom_sim.pth')
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


#Plot some training image
real_batch = next(iter(dataloaderB))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=1, normalize=True).cpu(), (1,2,0)))
plt.savefig(os.path.join(args.result_path, "real_sample.png"))


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


checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
netE = AE.encoder_(args).cuda()
netDA = AE.decoder_(args).cuda()
netDB = AE.decoder_(args).cuda()
modelA = AE.autoencoder(args, netE, netDA)
modelB = AE.autoencoder(args, netE, netDB)

modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelA.to(device)

modelB.load_state_dict(checkpoint['modelB_state_dict'])
modelB.to(device)

real_img = real_batch[0].to(device)[:32]
#real_img = torch.unsqueeze(real_img, 0)
out_imgA = modelA(real_img)
out_imgB = modelB(real_img)
print("saving.....")
save_image(out_imgA, './fake_imgB_from_modelA.png')
save_image(out_imgB, './fake_imgB_from_modelB.png')
save_image(real_img, './real_img.png')
print("save predict_img success!")
