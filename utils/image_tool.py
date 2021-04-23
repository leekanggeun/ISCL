import numpy as np
import os
from PIL import Image

def normalize(data, mean, std):
    return (data - mean)/std

def denormalize(data, mean, std):
    return (data * std) + mean

def gaussian_noise(img, mean, std):
    # The range of image must be [0, 255]
    assert img.ndim == 3 or img.ndim == 2, ("Check the dimension of input")
    gauss = np.random.normal(mean, std, img.shape)
    return np.array(img+gauss, dtype=np.float32)
    
def salt_and_pepper_noise(img, density):
    assert img.ndim == 3 or img.ndim == 2, ("Check the dimension of input")
    sp_ratio = 0.5 # ratio between salt and pepper
    n_salt = int(img.size*density*sp_ratio)
    n_pepper = int(img.size*density*(1-sp_ratio))
    noisy = np.copy(img) 
    idx = [np.random.randint(0, length-1, n_salt) for length in img.shape]
    noisy[idx] = 1
    idx = [np.random.randint(0, length-1, n_pepper) for length in img.shape]
    noisy[idx] = 0
    return noisy

def speckle_noise(img, std):
    assert img.ndim == 3 or img.ndim == 2, ("Check the dimension of input")
    assert std <= 1 and std >= 0, ("Standard deviation should be in the range [0,1]")
    noise = np.random.uniform(((12*std)**2)*-.5, ((12*std)**2)*.5, img.shape)
    return np.array(img+noise*img, dtype=np.float32)

def image_read(path):
    output = []
    if ('.tif' not in path) and ('.png' not in path):
        tif_list = sorted(os.listdir(path))
        for name in tif_list:
            im = Image.open(path+"/"+name)
            output.append(np.array(im, dtype=np.float32))
    else:
        im = Image.open(path)
        output = np.array(im, dtype=np.float32)
    return output

def image_division(image, patch_size):
    assert (image.ndim <= 4 and image.ndim >= 3), ("Check the dimension of inputs") # n, h, w or n, h, w, 1
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)
        
    n = len(image)
    patch_x, patch_y = patch_size
    output = []
    for i in range(0,n):
        temp = image[i]
        x,y,z = np.shape(temp)
        p = int(np.ceil(x/patch_x))  
        q = int(np.ceil(y/patch_y)) 
        for j in range(0,p):
            for k in range(0,q):
                if j == p-1:
                    if k == q-1:
                        output.append(temp[-patch_x:,-patch_y:,0:z])
                    else:
                        output.append(temp[-patch_x:,k*patch_y:(k+1)*patch_y,0:z])
                else:
                    if k == q-1:
                        output.append(temp[j*patch_x:(j+1)*patch_x,-patch_y:,0:z])
                    else:
                        output.append(temp[j*patch_x:(j+1)*patch_x,k*patch_y:(k+1)*patch_y,0:z])

    return np.array(output, dtype=np.float32)

def image_augmentation(x):
    assert (x.ndim == 4 or x.ndim == 3), ("Check the dimension of inputs")
    if x.ndim==3:
        n,w,h = np.shape(x)[0:3]
        out = np.zeros([n*8,w,h], dtype=np.float32)
        for f in range(0,2):
            for r in range(0,4):
                if f == 0 and r == 0:
                    out[:n] = x
                    continue
                for i in range(0,n):
                    out[i+(n*r)+f*4*n] = np.flip(np.rot90(x[i],r),f)
    elif x.ndim==4:
        n,w,h,z = np.shape(x)[0:4]
        out = np.zeros([n*8,w,h,z], dtype=np.float32)
        for f in range(0,2):
            for r in range(0,4):
                if f == 0 and r == 0:
                    out[:n] = x
                    continue
                for i in range(0,n):
                    for j in range(0,z):
                        out[i+(n*r)+f*4*n,:,:,j] = np.flip(np.rot90(x[i,:,:,j],r),f)
    return out

