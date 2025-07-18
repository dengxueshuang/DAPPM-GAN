# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:59:34 2022

trying loss2 > loss 1 : done

@author: Achintha

This code uses 1% from oneside 10% from the other side
use loadersUNETCGAN_f woth fix_sample = 1
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
from torchvision.utils import save_image
from lib import loaders, modules, loss
from loss.loss import Style,Perceptual
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from lib.EncoderModels import ResnetGenerator, Discriminator
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import scipy.stats as stats

TV_WEIGHT = 1e-7

def tvloss(y_hat):
    diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    tv_loss = TV_WEIGHT*(diff_i + diff_j)
    return tv_loss

def gradient_img(img,device):
    img = img.squeeze(0)
    a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device))
    G_x=conv1(img)
    b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).to(device))
    G_y=conv2(img)
    
    c=np.array([[-2, -1, -0],[-1,0,1],[0,1,2]])
    conv3=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv3.weight=nn.Parameter(torch.from_numpy(c).float().unsqueeze(0).unsqueeze(0).to(device))
    G_xy=conv3(img)#conv1(Variable(x)).data.view(1,x.shape[2],x.shape[3])

    d=np.array([[0, 1, 2],[-1,0,1],[-2,-1,0]])
    conv4=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv4.weight=nn.Parameter(torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device))
    G_yx=conv4(img)#(Variable(x)).data.view(1,x.shape[2],x.shape[3])
    
    G = torch.cat([G_x,G_y,G_xy,G_yx],dim=1)
    #G = torch.cat([G_x,G_y,G_xy,G_yx],dim=1)

    #G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G

def power_loss(images,device,E=100):
    #print('images: ',images.shape)
    image = images.detach().cpu().numpy().astype(int)
    npix = images.shape[1]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    
    kfreq = np.fft.fftfreq(npix) * npix
    
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.reshape(images.shape[0],-1)
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    ind = np.argpartition(Abins, E)
    return torch.FloatTensor(ind).to(device)
    
    
    

def calc_loss_dense(pred, target, metrics):
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def calc_loss_sparse(pred, target, samples, metrics, num_samples):
    criterion = nn.MSELoss()
    loss = criterion(samples*pred, samples*target)*(256**2)/num_samples
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))

def concat_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)
    return combined

def ohe_vector_from_labels(labels, n_classes):
    return F.one_hot(labels, num_classes=n_classes)

#风格损失


class GANTrain():
    def __init__(self,netD,netG,trainset,valset= [],testset =[],phase='first',batchsize = 15, experiment_path=''):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.batch_sz = batchsize
        self.n_segments = 100
        self.c = 100
        self.phase = phase
        self.trainset = trainset
        self.testset = testset
        self.valset = valset
        self.lossD = nn.BCEWithLogitsLoss()
        self.lossG = nn.MSELoss()#nn.L1Loss()#nn.MSELoss()
        self.lossGS = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.lossMsSSIM = loss.MS_SSIM_L1_LOSS()
        self.lossL1 = nn.L1Loss()
        self.netG = netG.to(self.device)#ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6).to(self.device)
        self.netD = netD.to(self.device)#Discriminator(self.device).to(self.device)
        self.optimG = torch.optim.Adam(self.netG.parameters(),lr=0.001, betas=(0.9, 0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(),lr=0.001, betas=(0.9, 0.999))
        self.train_loader = torch.utils.data.DataLoader(self.trainset,batch_size=self.batch_sz,shuffle=False,num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(self.testset,batch_size=self.batch_sz,shuffle=False,num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(self.valset,batch_size=self.batch_sz,shuffle=False,num_workers=2)
        self.experiment_path = experiment_path
        
    def  model_train(self,epochs=10):
        self.lossDL = []
        self.lossGL = []
        self.lossMSE = []
        self.lossDL_m = []
        self.lossGL_m = []
        self.lossMSE_m = []
        self.val_loss_m = []
        best_model_wts_G = copy.deepcopy(self.netG.state_dict())
        best_model_wts_D = copy.deepcopy(self.netD.state_dict())
        best_loss = 1000
        i = 0
        # Dense training
        for epoch in tqdm(range(epochs),unit='epochs'):
            if epoch%25==0:
                for g in self.optimG.param_groups:
                    g['lr'] = g['lr'] /10
                for d in self.optimD.param_groups:
                    d['lr'] = d['lr'] /10
            print('learning rate gen:',g['lr'] ,' learning rate dis:',d['lr'])
            for number,(inps, gts) in enumerate(self.train_loader):
                inps, gts = inps.to(self.device),gts.to(self.device)

                
                self.netG.train()
                self.netD.train()
                
                ############################
                # Train the Discriminator  #
                ############################
                
                one_hot_labelsF = ohe_vector_from_labels(torch.tensor([0]*self.batch_sz).to(self.device),2)
                one_hot_labelsR = ohe_vector_from_labels(torch.tensor([1]*self.batch_sz).to(self.device),2)
#                 print(one_hot_labelsF) # [1,2]
#                 print(one_hot_labelsR)
#                 print(one_hot_labelsR.shape)

                image_one_hot_labelsF = one_hot_labelsF[:, :, None, None]
                image_one_hot_labelsR = one_hot_labelsR[:, :, None, None]
#                 print(image_one_hot_labelsF.shape)

                self.netD.zero_grad()
                self.optimD.zero_grad()
                
                
                image_one_hot_labelsF = image_one_hot_labelsF.repeat(1, 1, inps.shape[2], inps.shape[3])
                image_one_hot_labelsR = image_one_hot_labelsR.repeat(1, 1, inps.shape[2], inps.shape[3])
#                 print(image_one_hot_labelsF.shape)
                #print()

                fake = self.netG(inps)
                fake_image_and_labels = concat_vectors(fake,image_one_hot_labelsF)
                real_image_and_labels = concat_vectors(gts, image_one_hot_labelsR)
                #print('fake shape:',fake_image_and_labels.shape)
#                 df

                predD_fake = self.netD(fake_image_and_labels.detach())
                predD_real = self.netD(real_image_and_labels)
                #print(predD_fake.shape)
                # print(predD_real.shape)
                lossD_fake = self.lossD(predD_fake,torch.zeros_like(predD_fake))
                lossD_real = self.lossD(predD_real,torch.zeros_like(predD_real))

                lossDT = (lossD_fake + lossD_real)/2

                lossDT.backward(retain_graph=True)
                self.optimD.step()

                self.lossDL += [lossDT.data.cpu().numpy()/inps.shape[0]]
                #print("Discriminator loss: ",lossDT.item())

                #####################
                # Train Generator   #
                #####################

                self.netG.zero_grad()
                self.optimG.zero_grad()

                fake = self.netG(inps)

                fake_image_and_labels = concat_vectors(fake,image_one_hot_labelsR)
                predD_fake = self.netD(fake_image_and_labels)
                
                # print('fake: ',torch.unique(torch.isnan(fake)))
                # print('inps: ',torch.unique(torch.isnan(inps)))

                if self.phase == 'first':
                    lossG = self.lossD(predD_fake,torch.ones_like(predD_fake))
                    style_loss = Style(fake,gts)
                    perceptual_loss = Perceptual(fake,gts)
                    #lossM = self.lossG(fake,gts)
                    g_fake = gradient_img(fake,self.device)
                    #g_fake = torch.nan_to_num(g_fake)#,nan=0,posinf=100,neginf=100)
                    #g_fake = gradient_img(fake,self.device)
                    #g_fake = torch.nan_to_num(g_fake)#,nan=0,posinf=100,neginf=100)
                    #g_targets = gradient_img(gts, self.device)
                    #lossGS = 1 - torch.mean(self.lossGS(g_targets, g_fake))
                    ##lossTV = tvloss(fake)
                    # print('Lg: ',lossG)
                    # print('lm: ',lossM)
                    # print('lGS: ',lossGS)
                    # print('ltv: ',lossTV)
    
                    #lossT = 10*lossG + 1*lossM + 10*lossTV + 10*lossGS#+ lossTV
                    lossT = 10 * lossG +250*style_loss +0.1*perceptual_loss
                    

                

                lossT.backward()
                
                self.optimG.step()

                self.lossGL += [lossT.data.cpu().numpy()/inps.shape[0]]
                with torch.no_grad():
                    loss = self.lossG(fake,gts)
                    self.lossMSE += [loss.data.cpu().numpy()]
                    if number%100 == 0:
                        print("epoch: ",epoch,' number: ',number,' MSE: ',np.mean(self.lossMSE))
                #print("Generator loss: ",lossG.item())
            #     break
            # break
                #print(lossT)
            print("mean D loss: ", np.mean(np.array(self.lossDL)))
            print("mean G loss: ",np.mean(np.array(self.lossGL)))
            print("mean MSE loss: ",np.mean(np.array(self.lossMSE)))
            self.lossDL_m += [np.mean(np.array(self.lossDL))]
            self.lossGL_m += [np.mean(np.array(self.lossGL))]
            self.lossMSE_m += [np.mean(np.array(self.lossMSE))]
            
            with torch.no_grad():
                self.netD.eval()
                self.netG.eval()
                self.val_loss = []
                for inps,gts in self.val_loader:
                    inps, gts = inps.to(self.device),gts.to(self.device)
                    fake = self.netG(inps)
                    v_loss = self.lossG(fake,gts)
                    self.val_loss += [v_loss.item()]
                val = np.mean(np.array(self.val_loss))
                self.val_loss_m += [val]
                if val < best_loss:
                    best_loss = val
                    print("saving best model")
                    print("val MSE: ",val)
                    best_model_wts_D = copy.deepcopy(self.netD.state_dict())
                    best_model_wts_G = copy.deepcopy(self.netG.state_dict())
                    G_file = os.path.join(self.experiment_path, f'Trained_ModelMSE_Gen_best_{i}.wgt')
                    D_file = os.path.join(self.experiment_path, f'Trained_ModelMSE_Dis_best_{i}.wgt')
                    torch.save(self.netG.state_dict(), G_file)
                    torch.save(self.netD.state_dict(), D_file)
                    i+=1
            np.savetxt(os.path.join(self.experiment_path, "MSE_train.csv"), self.lossMSE_m, delimiter=",")
            np.savetxt(os.path.join(self.experiment_path, "MSE_val.csv"), self.val_loss_m, delimiter=",")
            
            # 保存最新模型状态
            torch.save(best_model_wts_D, os.path.join(self.experiment_path, 'Latest_Trained_ModelMSE_D.pt'))
            torch.save(best_model_wts_G, os.path.join(self.experiment_path, 'Latest_Trained_ModelMSE_G.pt')) 
        return best_model_wts_D,best_model_wts_G
                    
        #torch.save(model.state_dict(), 'RadioWNet_c_DPM_Thr2/Trained_Model_FirstU.pt')

########################
# Load dataset         #
########################

# Setup 1 - uniform 1%
# Setup 2 - twoside 1% and 10%
# Setup 3 - nonuniform 1% to 10%

setup = 1       # REVISE index of setup
setups = ['uniform', 'twoside', 'nonuniform']
setup_name = setups[setup-1]

if setup == 1:          
    Radio_train = loaders.RadioUNet_s(phase="train", fix_samples=655, num_samples_low=10, num_samples_high=300)
    Radio_val = loaders.RadioUNet_s(phase="val", fix_samples=655, num_samples_low=10, num_samples_high=300)
    Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=655, num_samples_low=10, num_samples_high=300)
    
elif setup == 2:
    Radio_train = loaders.RadioUNet_s(phase="train", fix_samples=1, num_samples_low=655, num_samples_high=655*10)
    Radio_val = loaders.RadioUNet_s(phase="val", fix_samples=1, num_samples_low=655, num_samples_high=655*10)
    Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=1, num_samples_low=655, num_samples_high=655*10)
    
else:
    Radio_train = loaders.RadioUNet_s(phase="train", fix_samples=0, num_samples_low=655, num_samples_high=655*10)
    Radio_val = loaders.RadioUNet_s(phase="val", fix_samples=0, num_samples_low=655, num_samples_high=655*10)
    Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=0, num_samples_low=655, num_samples_high=655*10)
    
image_datasets = {
    'train': Radio_train, 'val': Radio_val
}

batch_size = 4


dataloaders = {
    'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1),
    'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
}            

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

exp_index = 1           # REVISE index of experiment
exp_path = f"{setup_name}_{exp_index}"

# Ensure the directory exists before saving
os.makedirs(exp_path, exist_ok=True)

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled
#netG = ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6)
netG = modules.RadioWNet(phase="firstU")#.to(device)
#netG.load_state_dict(torch.load('UNetCGANDEBUG/Trained_ModelMSE_G2.pt'))
netD = Discriminator(device)
#netD.load_state_dict(torch.load('UNetCGANDEBUG/Trained_ModelMSE_D2.pt'))

RadioGAN = GANTrain(netD,netG,Radio_train,Radio_val,Radio_test,phase='first',experiment_path=exp_path)
if __name__ == '__main__':
    bestD, bestG = RadioGAN.model_train(epochs=100)
    #np.savetxt(os.path.join(exp_path, "MSE_train.csv"), RadioGAN.lossMSE_m ,delimiter=",")
    #np.savetxt(os.path.join(exp_path, "MSE_val.csv"), RadioGAN.val_loss_m, delimiter=",") 
    torch.save(bestD, os.path.join(exp_path, 'Trained_ModelMSE_D.pt'))
    torch.save(bestG, os.path.join(exp_path, 'Trained_ModelMSE_G.pt'))                
