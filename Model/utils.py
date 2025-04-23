import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import Tensor



import matplotlib.pyplot as plt
from unet import UNet



def get_model(name, parts, device):
    if name == "UNet":
        # model = UNet(n_channels=len(b_values_no0), n_classes=8, bilinear=False).to(device)

        # modified
        model = UNet(n_channels=7, n_classes=parts, bilinear=False).to(device)

    return model

def set_seed(seed=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_normal(m):
    if type(m) == nn.Linear:
        # nn.init.uniform_(m.weight)
        # nn.init.kaiming_uniform_(m.weight)
        nn.init.xavier_uniform_(m.weight)
        
        
def descale_params(paramnorm, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return ((((paramnorm - a1) / (b1 - a1)) ) * (Upper_bound - Lower_bound)) + Lower_bound


def scaling(param, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return (b1 - a1) * ((param - Lower_bound) / (Upper_bound - Lower_bound)) + a1


def Descale_params(params, bounds):
    S0_descaled = descale_params(params[:,0],bounds[0,0], bounds[1,0])  
    D_descaled = descale_params(params[:,1],bounds[0,1], bounds[1,1])   #Dt
    F_descaled = descale_params(params[:,2],bounds[0,2], bounds[1,2])   #Fp
    Dp_descaled = descale_params(params[:,3],bounds[0,3], bounds[1,3])  #Dp
    params_descaled = torch.cat((S0_descaled[:,None], D_descaled[:,None], F_descaled[:,None], Dp_descaled[:,None]), axis=1)
    return params_descaled


def Scale_params(params, bounds):
    S0_scaled = scaling(params[:,0],bounds[0,0], bounds[1,0])
    D_scaled = scaling(params[:,1],bounds[0,1], bounds[1,1])  #Dt
    F_scaled = scaling(params[:,2],bounds[0,2], bounds[1,2])  #Fp
    Dp_scaled = scaling(params[:,3],bounds[0,3], bounds[1,3]) #Dp
    params_scaled = torch.cat((S0_scaled[:,None], D_scaled[:,None], F_scaled[:,None], Dp_scaled[:,None]), axis=1)
    return params_scaled

# bi-exponential function
def funcBiExp(b, f, Dt, Ds):
    ## Units
    # b: s/mm^2
    # D: mm^2/s
    return f * np.exp(-1.*Ds * b) + (1.-f) * np.exp(-1.*Dt * b)

def ivim(b, Dp, Dt, Fp):
    return Fp*np.exp(-b*Dp) + (1-Fp)*np.exp(-b*Dt)


    
    

    
class BreastDataset_seg(Dataset):
    def __init__(self, time_point, root_dir="", train_val = "train",denoise=False, \
                one_or_twoDim=True, seg_num = 4, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        # self.all_file_path = []
        self.fname_gt ='_IVIMParam.npy'
        self.fname_tissue ='_TissueType.npy'
        self.fname_noisyDWIk = '_NoisyDWIk.npy'
        self.fname_DWIk = '_gtDWIs.npy'
        # self.num_data = 1000
        self.train_val = train_val
        self.index = np.linspace(0,999,1000)
        if self.train_val == "train":
            self.data_index = self.index[:700]
        elif self.train_val == "val":
            self.data_index = self.index[700:900]
        elif self.train_val == "inf":
            self.data_index = self.index[900:1000]
            print("self.data_index",self.data_index)
        self.time_point = time_point
        self.one_or_twoDim = one_or_twoDim
        self.denoise = denoise

        self.transform = transform
        self.num_tissue_seg = seg_num

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = int(self.data_index[idx])
        x = read_data(self.root_dir, self.fname_gt, idx+1)
        # NoisyDWIk= read_data(self.root_dir, self.fname_noisyDWIk, idx+1)

        # import the data, and apply ifft to convert from K space to spatial space
        NoisyDWIk= read_data(self.root_dir, self.fname_noisyDWIk, idx+1)
        arr3D_img = np.abs(np.fft.ifft2(NoisyDWIk, axes=(0,1) ,norm='ortho'))
        tissue_img= read_data(self.root_dir, self.fname_tissue, idx+1)

        # merge the classification, and can be modified
        if self.num_tissue_seg == 3:
            tissue_img = tissue_img - 1
            tissue_img[tissue_img == 1] = 1
            tissue_img[tissue_img == 2] = 1
            tissue_img[tissue_img == 3] = 1
            tissue_img[tissue_img == 4] = 1
            tissue_img[tissue_img == 5] = 1
            tissue_img[tissue_img == 6] = 1
            tissue_img[tissue_img == 7] = 2  # modified
        if self.num_tissue_seg == 4:
            tissue_img = tissue_img - 1
            tissue_img[tissue_img == 1] = 1
            tissue_img[tissue_img == 2] = 1
            tissue_img[tissue_img == 3] = 1
            tissue_img[tissue_img == 4] = 1
            tissue_img[tissue_img == 5] = 1
            tissue_img[tissue_img == 6] = 2
            tissue_img[tissue_img == 7] = 3  # modified
        if self.num_tissue_seg == 8:
            tissue_img = tissue_img - 1

        if self.denoise:
            kernelSize = 3
            kernel = (kernelSize, kernelSize)
            stake = []
            for layer in range(arr3D_img.shape[2]):
                noise_slides = arr3D_img[:,:,layer]
                # print(noise_slides.shape)
                noise_slides_de = denoise_gaussian(noise_slides, kernel)
                stake.append(noise_slides_de)
            arr3D_img = np.stack(stake, axis=2)
        # Normalize the input MRI, divided by the layer b=0
        if self.time_point==7:
            arr3D_img = arr3D_img[:,:,1:] / (arr3D_img[:,:,0][:,:,np.newaxis])
        else:
            arr3D_img = arr3D_img[:,:,:]# / (arr3D_img[:,:,0][:,:,np.newaxis])

        

        if self.one_or_twoDim:
            x = x.reshape([-1,3])
            arr3D_img = arr3D_img.reshape([-1,self.time_point])
        else:
            x = x.transpose([2,0,1])
            arr3D_img = arr3D_img.transpose([2,0,1])
        preprocessed = arr3D_img
        if self.transform != None:
            preprocessed = self.transform(preprocessed)
        
        return preprocessed,tissue_img




# class BreastDataset_refine(Dataset):
#     def __init__(self, time_point, root_dir, nninfer_dir, train_val = True,denoise=False, \
#                 one_or_twoDim=True):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.root_dir = root_dir
#         self.nninfer_dir = nninfer_dir
#         self.all_file_path = []
#         self.fname_gt ='_IVIMParam.npy'
#         self.fname_tissue ='_TissueType.npy'
#         self.fname_noisyDWIk = '_NoisyDWIk.npy'
#         self.fname_DWIk = '_gtDWIs.npy'
#         self.nn_infer = '.npy'
        
#         self.num_data = 1000
#         self.train_val = train_val
#         self.file_list = os.listdir(root_dir)
#         self.data_num = len(self.file_list)//4
#         self.index = np.linspace(0,999,1000)
#         if self.train_val:
#             self.data_index = self.index[:950]
#         else:
#             self.data_index = self.index[950:]
#             print("self.data_index",self.data_index)
#         self.time_point = time_point
#         self.one_or_twoDim = one_or_twoDim
#         self.denoise = denoise

#     def __len__(self):
#         return len(self.data_index)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         idx = int(self.data_index[idx])
#         gt = read_data(self.root_dir, self.fname_gt, idx+1)
#         # NoisyDWIk= read_data(self.root_dir, self.fname_noisyDWIk, idx+1)
#         nninfer = read_data(self.nninfer_dir, self.nn_infer, idx+1)
#         nninfer = np.transpose(nninfer,(2,0,1))
#         gt = np.transpose(gt,(2,0,1))

#         # print("gt_t",gt_t.shape,"arr3D_img",arr3D_img.shape,"x",x.shape)
#         # image = image /np.max(image) 
#         # print("image",image.size)
#         # print("image",image.shape)
#         return nninfer,gt


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask


    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean(), dice



def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)[0]

# this function unzips the target file to the destination path. need to use 'import zipfile'
def unzip_data(target_dir, des_dir):
    zipfile.ZipFile(target_dir, 'r').extractall(des_dir)
   
# this function reads an npy file for case i
# need to use 'import  numpy as np'
def read_data(file_dir, fname, i):
    fname_tmp = file_dir + "{:04}".format(i) + fname
    data = np.load(fname_tmp)
    return data

# this fucntion computes the rRMSE for one microstructual parameter image for each case
# y is the reference solution
# t is the tissue type image
def rRMSE_D(x,y,t):
    
    Nx = x.shape[0]
    Ny = x.shape[1]
    
    t_tmp = np.reshape(t, (Nx*Ny,))
    tumor_indice = np.argwhere(t_tmp == 8)
    non_tumor_indice = np.argwhere(t_tmp != 8)
    non_air_indice = np.argwhere(t_tmp != 1)
    non_tumor_air_indice= np.intersect1d(non_tumor_indice,non_air_indice)
    
    x_tmp = np.reshape(x, (Nx*Ny,))
    x_t = x_tmp[tumor_indice]
    x_nt = x_tmp[non_tumor_air_indice]
    
    y_tmp = np.reshape(y, (Nx*Ny,))
    y_t = y_tmp[tumor_indice]
    y_nt = y_tmp[non_tumor_air_indice]
    
    # tumor region
    tmp1 = np.sqrt(np.sum(np.square(y_t)))
    tmp2 = np.sqrt(np.sum(np.square(x_t-y_t)))
    z_t = tmp2 / tmp1
    
    # non-tumor region
    tmp1 = np.sqrt(np.sum(np.square(y_nt)))
    tmp2 = np.sqrt(np.sum(np.square(x_nt-y_nt)))
    z_nt = tmp2 / tmp1
    
    return z_t, z_nt

def rRMSE_f(x,y,t):

    Nx = x.shape[0]
    Ny = x.shape[1]
    
    t_tmp = np.reshape(t, (Nx*Ny,))
    tumor_indice = np.argwhere(t_tmp == 8)
    non_tumor_indice = np.argwhere(t_tmp != 8)
    non_air_indice = np.argwhere(t_tmp != 1)
    non_tumor_air_indice= np.intersect1d(non_tumor_indice,non_air_indice)
    
    x_tmp = np.reshape(x, (Nx*Ny,))
    x_t = x_tmp[tumor_indice]
    x_nt = x_tmp[non_tumor_air_indice]
    
    y_tmp = np.reshape(y, (Nx*Ny,))
    y_t = y_tmp[tumor_indice]
    y_nt = y_tmp[non_tumor_air_indice]
    
    # tumor region
    tmp1 = np.sqrt(tumor_indice.shape[0])
    tmp2 = np.sqrt(np.sum(np.square(x_t-y_t)))
    z_t = tmp2 / tmp1
    
    # non-tumor region
    tmp1 = np.sqrt(non_tumor_air_indice.shape[0])
    tmp2 = np.sqrt(np.sum(np.square(x_nt-y_nt)))
    z_nt = tmp2 / tmp1
    
    return z_t, z_nt

# this fucntion computes the rRMSE for one case
def rRMSE_per_case(x_f,x_dt,x_ds,y_f,y_dt,y_ds,t):
    R_f_t, R_f_nt = rRMSE_f(x_f, y_f, t)
    R_Dt_t, R_Dt_nt = rRMSE_D(x_dt, y_dt, t)
    R_Ds_t, R_Ds_nt = rRMSE_D(x_ds, y_ds, t)
    
    z =  (R_f_t + R_Dt_t + R_Ds_t)/3 + (R_f_nt + R_Dt_nt)/2
    z_t =  (R_f_t + R_Dt_t + R_Ds_t)/3
    
    return z, z_t

# this fucntion computes the rRMSE for all cases
# y is the reference solution
def rRMSE_all_cases(x_f,x_dt,x_ds,y_f,y_dt,y_ds,t):
    z = np.empty([x_f.shape[2]])
    z_t = np.empty([x_f.shape[2]])
    for i in range(x_f.shape[2]):
        z[i], z_t[i] = rRMSE_per_case(x_f[:,:,i],x_dt[:,:,i],x_ds[:,:,i],y_f[:,:,i],y_dt[:,:,i],y_ds[:,:,i],t[:,:,i]) 
        
    return np.average(z), np.average(z_t)

if __name__=="__main__":
    breast_data = BreastDataset_seg("/proj/berzelius-2023-99/users/x_zyang/aapm2024/train/")
    trainloader = utils.DataLoader(breast_data,
                                    batch_size = 64, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)    
    for img,lable in trainloader:
        print("img.shape",img.shape,"lable.shape",lable.shape)
        img = img.reshape([-1,8])
        lable = lable.reshape([-1,3])
        print("img.shape",img.shape,"lable.shape",lable.shape)        
        print("img.shape",np.unique(np.abs(img)),"lable.shape",lable.shape,np.unique(lable))        
