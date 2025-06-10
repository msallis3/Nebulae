from __future__ import annotations


from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from photutils.datasets import make_noise_image
from astropy.stats import sigma_clip
import glob
from scipy.stats import scoreatpercentile
from astropy.modeling import models, fitting
from numpy import ma


def create_median_bias():
    input_dir = Path("Data")
    output_file = "ring_bias.fits"
    
    bias_images = [] #reading all the files
    files = sorted(glob.glob('hstu/work/3-Crabs/Data/Bias_BIN1_20250603_*.fits')) #I included the full path, not sure if it is all necessary
    
    for file in files:
        image = fits.getdata(file)[1000:1500, 2000:2500].astype('f4') #converting to numpyfloat32 format
        bias_images.append(image)

    sig_clip = sigma_clip(bias_images, cenfunc='median', sigma=3.0, axis=0)

    median_bias = np.median(sig_clip, axis = 0)
    hdul = fits.PrimaryHDU(data=median_bias.data) #saving
    hdul.writeto(ring_bias, overwrite=True)   #saving with the proper filename


    #plotting (not originally in hw), copied from twinjets code so they are easier to compare
    plt.imshow(median_bias, origin='lower', cmap='gray', vmin=np.percentile(median_bias, 5), vmax=np.percentile(median_bias, 95))
    plt.colorbar(label='Counts')
    plt.savefig("Bias.png", dpi=150)
    plt.show()
    
    return median_bias.data