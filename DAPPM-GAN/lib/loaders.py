from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")


class RadioUNet_s(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)
    And we assume a fixed sample size of 1% of 256x256"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="E:/DeepLearning/REMMap/RadioMapSeer/",     # path to dataset
                 numTx=80,
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=655*10,        
                 num_samples_low= 655, 
                 num_samples_high= 655*10,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """
        

        
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        self.arr = np.arange(256)
        self.one = np.ones(256)
        self.img = np.outer(self.arr,self.one) 

        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
                
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
            
        image_gain = image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                 
        #input measurements
        image_samples = np.zeros((256,256))
        if self.fix_samples==0:
            num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_samples=np.floor(self.fix_samples).astype(int)               
        x_samples=np.random.randint(0, 255, size=num_samples)
        y_samples=np.random.randint(0, 255, size=num_samples)
        
        if self.fix_samples ==1:
            side = np.random.randint(0,2)
            if side == 1:
                x_samples = np.append(np.random.randint(0,128,size=6550),np.random.randint(128,255,size=655))
                y_samples = np.random.randint(0,255,size=6550+655)
            else:
                x_samples = np.append(np.random.randint(0,128,size=655),np.random.randint(128,255,size=6550))
                y_samples = np.random.randint(0,255,size=6550+655)
                
        image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]



        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_samples], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        return [inputs, image_gain]