import torch
from torch.utils.data.dataset import Dataset
from preprocessing import *
import numpy as np
import random
from random import randint


class SEMDataTrain(Dataset):

    def __init__(self, image, mask):

        self.mask_arr = mask
        self.image_arr = image
        self.data_len = len(self.mask_arr)

    def __getitem__(self, index):
        img_as_np = self.image_arr[index]
        
        
        flip_num = randint(0, 3)
        img_as_np = flip(img_as_np, flip_num)        

        if randint(0, 1):
            # Gaussian_noise
            gaus_sd, gaus_mean = randint(0, 20), 0
            img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
        else:
            # uniform_noise
            l_bound, u_bound = randint(-20, 0), randint(0, 20)
            img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)
                

        # Brightness
        pix_add = randint(-20, 20)
        img_as_np = change_brightness(img_as_np, pix_add)

        # Normalize the image
        img_as_np = normalization2(img_as_np, max=1, min=0)
        # img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        msk_as_np = self.mask_arr[index]
        
        msk_as_np = flip(msk_as_np, flip_num)
        msk_as_np = np.squeeze(msk_as_np)
        # msk_as_np = msk_as_np/255


        msk_as_tensor = torch.from_numpy(msk_as_np.copy())  # Convert numpy array to tensor

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len

class SEMDataVal(Dataset):
    def __init__(self, image, mask):
        self.mask_arr = mask
        self.image_arr = image
        self.data_len = len(self.mask_arr)
        
    def __getitem__(self, index):

        img_as_np = self.image_arr[index]

        img_as_np = normalization2(img_as_np, max=1, min=0)

        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        msk_as_np = self.mask_arr[index]
        msk_as_np = np.squeeze(msk_as_np)
        # msk_as_np = msk_as_np/255

        msk_as_tensor = torch.from_numpy(msk_as_np)  # Convert numpy array to tensor

        return (img_as_tensor, msk_as_tensor)
    
    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


#################### OLD LOADER WITH COMMON TRAIN, VAL LOADER#######################################

# class SEMDataTrain(Dataset):

#     def __init__(self, image, mask):

#         self.mask_arr = mask
#         self.image_arr = image
#         self.data_len = len(self.mask_arr)

#     def __getitem__(self, index):
#         img_as_np = self.image_arr[index]
        
#         # flip_num = randint(0, 3)
#         # img_as_np = flip(img_as_np, flip_num)        

#         if randint(0, 1):
#             # Gaussian_noise
#             gaus_sd, gaus_mean = randint(0, 20), 0
#             img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
#         else:
#             # uniform_noise
#             l_bound, u_bound = randint(-20, 0), randint(0, 20)
#             img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)
                

#         # Brightness
#         pix_add = randint(-20, 20)
#         img_as_np = change_brightness(img_as_np, pix_add)

#         # Normalize the image
#         img_as_np = normalization2(img_as_np, max=1, min=0)
#         # img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
#         img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

#         msk_as_np = self.mask_arr[index]
#         msk_as_np = np.squeeze(msk_as_np)
        
#         # msk_as_np = flip(msk_as_np, flip_num)
#         # msk_as_np = msk_as_np/255

#         msk_as_tensor = torch.from_numpy(msk_as_np)  # Convert numpy array to tensor

#         return (img_as_tensor, msk_as_tensor)

#     def __len__(self):
#         """
#         Returns:
#             length (int): length of the data
#         """
#         return self.data_len