from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse
from skimage import data, img_as_float
import cv2
import time
import os
import numpy as np
from matplotlib import pylab as plt
import time
import math

def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

def img2np(dir=[],img_len=128):
    img=[]
    for x in dir:
        try:img.append(cv2.imread(x))
        except:continue
        if img_len!=0:img[-1]=cv2.resize(img[-1],(img_len,img_len))
        elif img[-1].shape!=img[0].shape:img.pop(-1);continue#Leave only the same shape
        img[-1] = img[-1].astype(np.float32)/ 256
    return np.stack(img, axis=0)

preds=[]
test=ffzk(os.path.join("./", 'datasets/div2k_srlearn/test_y'))

preds.append(ffzk('datasets/div2k_srlearn/test_cubic4'))
preds.append(ffzk('outputs/inception1_ngblur'))
preds.append(ffzk('outputs/inception1_cubic4'))
# preds.append(ffzk('outputs/unet3'))
# preds.append(ffzk('outputs/test4'))D
# preds.append(ffzk('outputs/ksvd5'))

max_sample_size=min([1000,len(test)])

if False:
    for i in range(len(preds)):
        psnrS=0.;ssimS=0.;mseS=0.
        targets="111.png"
        img1 = cv2.imread("/".join(test[0].split("/")[:-1])+"/"+targets)
        img2 = cv2.imread("/".join(preds[i][0].split("/")[:-1])+"/"+targets)
        mseS+=np.mean(np.square(img1.flatten().astype(np.float32)-img2.flatten().astype(np.float32)))
        print("MSE",mseS)
        print("PSNR",10.*math.log10((255.**2)/mseS))
        print("SSIM",ssim(img1, img2, multichannel=True))
        print("=S====^",i,"^=S====")
    exit()

for i in range(len(preds)):
    psnrS=0.;ssimS=0.;mseS=0.
    for ii in range(max_sample_size):
        img1 = cv2.imread(test[ii])
        img2 = cv2.imread(preds[i][ii])
        ssimS+=ssim(img1, img2, multichannel=True)
        mseS+=np.mean(np.square(img1.flatten().astype(np.float32)-img2.flatten().astype(np.float32)))
    psnrS/=max_sample_size;
    ssimS/=max_sample_size;
    mseS/=max_sample_size;
    print("MSE",mseS)
    print("PSNR",10.*math.log10((255.**2)/mseS))
    print("SSIM",ssimS)
    print("=====^",i,"^=====")
    
