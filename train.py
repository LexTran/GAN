from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np

from model import Generator,Discriminator

import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import torch.backends.cudnn as cudnn

import argparse
import os
import subprocess
import time
import math

# parameters
parser = argparse.ArgumentParser(description='DCGAN')
parser.add_argument('--resume_path', default=None, help='resume path')
parser.add_argument('--epoch', default=500, help='training epoch')
parser.add_argument('--loss', default='BCE', help='adversarial loss type')
parser.add_argument('--z_dim', default=100, help='dimension of latent z vector')
parser.add_argument('--ngf', default=64, help='dimension of generator feature map')
parser.add_argument('--ndf', default=64, help='dimension of discriminator feature map')
parser.add_argument('--bs', default=32, help='batch size')
parser.add_argument('--lr', default=0.01, help="learning rate")
parser.add_argument('--l1', default=1, help="lambda1 for adversarial loss")
parser.add_argument('--board', default='./runs', help="tensorboard path")
parser.add_argument('--save_path', default='./checkpoints/', help="save path")
parser.add_argument('--output_path', default='./output/', help="save epoch")
args = parser.parse_args()

# set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
    "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
cudnn.benchmark = False
device_ids = [1]

# loading datasets
batch_size = int(args.bs)
data_path1 = "./datasets/face/"

tfs = transforms.Compose([
    transforms.Resize(64),
    # transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

dataset = torchvision.datasets.ImageFolder(root=data_path1, transform=tfs)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

# model
model_g = Generator(int(args.z_dim), int(args.ngf))
model_d = Discriminator(int(args.ndf))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_g = model_g.to(device)
model_d = model_d.to(device)

# optimizer
d_optim = torch.optim.Adam(model_d.parameters(), lr=float(args.lr), betas=(0.5, 0.999))
g_optim = torch.optim.Adam(model_g.parameters(), lr=float(args.lr), betas=(0.5, 0.999))

# loss
if args.loss == 'BCE':
    adv_loss = nn.BCELoss()

# training
def train(epoch):
    writer = SummaryWriter(args.board)
    model_g.train()
    model_d.train()
    for i in range(epoch):
        total_g_loss = 0
        total_d_loss = 0
        fixed_z = torch.randn(1, int(args.z_dim), 1, 1).to(device)
        for image, _ in train_loader:
            # data prepare
            real_label = torch.ones(image.size(0), 1).to(device)
            fake_label = torch.zeros(image.size(0), 1).to(device)
            image = image.to(device)
            
            z = torch.randn(batch_size, int(args.z_dim), 1, 1).to(device)

            # train generator
            g_optim.zero_grad()
            generated_image = model_g(z)
            fake_out = model_d(generated_image)
            g_loss = adv_loss(fake_out, real_label)
            g_loss.backward()
            g_optim.step()

            # train discriminator
            d_optim.zero_grad()
            real_out = model_d(image)
            real_loss = adv_loss(real_out, real_label)
            fake_out = model_d(generated_image.detach())
            fake_loss = adv_loss(fake_out, fake_label)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optim.step()

            # record loss
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        writer.add_scalar('g_loss', total_g_loss/len(train_loader), i)
        writer.add_scalar('d_loss', total_d_loss/len(train_loader), i)
        print('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f', (i + 1, epoch, total_d_loss/len(train_loader), total_g_loss/len(train_loader)))

        # save image
        if (i+1) % 50 == 0:
            with torch.no_grad():
                # save checkpoints
                state = {
                    'generator': model_g.state_dict(),
                    'discriminator': model_d.state_dict(),
                    'epoch': epoch + 1
                }
                model_g.eval()
                image = model_g(fixed_z)
                writer.add_image('generated_image', image[0], i)
                torch.save(state, f'{args.save_path}/{i + 1}.pth') # save model and parameters
                print('Saving epoch %d model ...' % (i + 1))
                model_g.train()

train(int(args.epoch))