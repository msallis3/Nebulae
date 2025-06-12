from __future__ import annotations

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_body
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
import glob
from astropy.stats import sigma_clip
import os
import pathlib
import pytest
from photutils.datasets import load_star_image
import seaborn as sns

from astroscrappy import detect_cosmics

from astropy.table import Table
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.aperture import ApertureStats
import matplotlib.pyplot as plt

from pathlib import Path

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

def create_median_bias():
    input_dir = Path("Data")
    output_file = "twin_master_bias.fits"

    bias_data = []

    for file in sorted(input_dir.glob("Bias*.fits")):
        data = fits.getdata(file).astype('f4')

        bias_data.append(data) 
    if len(bias_data) == 0:
        raise RuntimeError(f"No bias")

    bias_3d = np.array(bias_data)

    clipping = sigma_clip(bias_3d, cenfunc='median', sigma=3, axis=0)
      
    median_bias = np.ma.median(clipping, axis=0)

    data_for_plot = median_bias.filled(np.nan)
    vmin = np.nanpercentile(data_for_plot, 5)
    vmax = np.nanpercentile(data_for_plot, 95)

    plt.imshow(data_for_plot, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Counts')
    plt.savefig(Path("Twin_Bias.png"), dpi=150)
    plt.show()
    
    output_path = Path("Twin_Images") / output_file
    primary = fits.PrimaryHDU(data=data_for_plot, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(output_path, overwrite=True)

    return data_for_plot

def create_median_dark():
    bias_filename = Path("Twin_Images") / "twin_master_bias.fits"
    output_file = Path('twin_master_dark.fits')
    bias_frame = fits.getdata(bias_filename).astype('f4')
    input_dir = Path("Data")

    dark_bias_data = []
    
    for f in sorted(input_dir.glob("Dark*.fits")):
        with fits.open(f) as dark:
            dark_data = dark[0].data.astype('f4')
            exptime = dark[0].header['EXPTIME']        
            dark_NObias = dark_data - bias_frame
            dark_bias_data.append(dark_NObias / exptime)
            header = dark[0].header.copy()
        
    one_dark = np.array(dark_bias_data)
    clipping_dark = sigma_clip(one_dark, cenfunc='median', sigma=3, axis=0)
    median_dark = np.ma.median(clipping_dark, axis=0)
    
    data_dark = median_dark.filled(np.nan)
    vmin = np.nanpercentile(data_dark, 5)
    vmax = np.nanpercentile(data_dark, 95)

    plt.imshow(data_dark, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Counts')
    plt.savefig(Path("Twin_Darks.png"), dpi=150)
    plt.show()
    
    output_path = Path("Twin_Images") / output_file
    primary = fits.PrimaryHDU(data=data_dark, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(output_path, overwrite=True)

    return data_dark

    
def create_median_flat():
    bias_filename = Path("Twin_Images") / "twin_master_bias.fits"
    dark_filename = Path("Twin Images") / "twin_master_dark.fits"
    
    output_file = Path("Twin_Images") / 'twin_master_flat.fits'
    input_dir = Path("Data")

    bias_frame = fits.getdata(bias_filename).astype('f4')
    dark_frame = fits.getdata(dark_filename).astype('f4')

    flat_data = []

    for fl in sorted(input_dir.glob('domeflat_*.fits')):
        with fits.open(fl) as flat:
            data = flat[0].data.astype('f4')
            sub_bias = data - bias_frame
            flat_data.append(sub_bias)

    array = np.array(flat_data)
    clipping = sigma_clip(array, cenfunc='median', sigma=3, axis=0)
    me_flat = np.ma.median(clipping, axis=0)
    median_flat = me_flat / np.ma.median(me_flat)
    median_flat = median_flat.filled(np.nan)


    vmin, vmax = np.percentile(median_flat, [1, 99])
    plt.imshow(median_flat, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Counts')
    plt.savefig(Path("Twin_Images") / "Twin_Flats.png", dpi=150)
    plt.show()
    
    
    primary = fits.PrimaryHDU(data=median_flat, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(output_file, overwrite=True)

    return median_flat

def reduce_science_frame():

    bias_filename = Path("Twin_Images") / "twin_master_bias.fits"
    dark_filename = Path("Twin_Images") / "twin_master_dark.fits"
    flat_filename = Path("Twin_Images") / "twin_master_flat.fits"
    
    output_file = Path("Twin_Images") / 'twin_science_reduced.fits'
    input_dir = Path("Data")

    bias_frame = fits.getdata(bias_filename).astype('f4')
    flat_frame = fits.getdata(flat_filename).astype('f4')
    dark_frame = fits.getdata(dark_filename).astype('f4')

    #Opening science stuff and trimming/float 32 
    for f in sorted(input_dir.glob('BFLY*.fits')):
        with fits.open(f) as science:
            data_s = science[0].data.astype('f4')

        #Getting the exposure time from header
            header = science[0].header
            exptime = science[0].header['EXPTIME']
  
    #Removing bias signal
    sub_bias = data_s - bias_frame

    #Subtracting dark frame with corrected exptime
    dark_corrected = sub_bias - dark_frame * exptime

    #Normalizing the flat
    flat_norm = flat_frame / np.mean(flat_frame)

    #Normalzing even more 
    corrected_science = dark_corrected / flat_norm
    
    #Getting rid of cosmic rays
    mask, cleaned = detect_cosmics(corrected_science)
    reduced_science = cleaned

    plt.imshow(reduced_science, cmap = 'magma', origin = 'lower', aspect = 'auto')
    plt.colorbar()
    plt.savefig(Path("Twin_Images") / 'Twin_reduced_science.png')
    plt.tight_layout()
    plt.close()
    

    #Saving
    primary = fits.PrimaryHDU(data=reduced_science, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(output_file, overwrite=True)

    return 


if __name__ == "__main__":
    science = reduce_science_frame()
    print('she ran')


#if __name__ == "__main__":
 #   photometry = do_aperture_photometry(positions, radii)
  #  print("It has printed")

def calculate_gain():
    output_file = Path("Twin_Images") / 'Twin_gain.fits'
    input_dir = Path("Data")
    

    flat1_path = input_dir / "domeflat_H-Alpha_001.fits"  # replace with actual filename
    flat2_path = input_dir / "domeflat_H-Alpha_002.fits"  # replace with actual filename

    flat1 = fits.getdata(flat1_path).astype(np.float32)
    flat2 = fits.getdata(flat2_path).astype(np.float32)

    
    bias_path = Path("Twin_Images") / "twin_master_bias.fits"
    bias = fits.getdata(bias_path).astype(np.float32)

    flat1 -= bias
    flat2 -= bias
    #Getting the difference of the two flats
    flat_diff = flat1 - flat2

    #Calculating variance 
    flat_var = np.var(flat_diff)

    #Getting average between the two
    mean = 0.5 * np.mean(flat1 + flat2)

    #Getting the gain with formula 
    gain = 2 * mean / flat_var

    print(f"Gain = {gain:.3f} e⁻/ADU")
    return gain
if __name__ == "__main__":
    gain = calculate_gain()

def calculate_readout_noise():

    output_file = Path("Twin_Images") / 'Twin_readout.fits'
    input_dir = Path("Data")
    
    bias1_path = input_dir / "Bias_BIN1_20250603_044940.fits"
    bias2_path = input_dir / "Bias_BIN1_20250603_044952.fits"

    bias1 = fits.getdata(bias1_path).astype('f4')
    bias2 = fits.getdata(bias2_path).astype('f4')

    gain = 0.501
    
    diff = bias1 - bias2
    bias_diff_var = np.var(diff)
    bias_diff_mean = np.mean(diff)
    bias_diff_std = np.std(diff)

    readout_noise_adu = np.sqrt(bias_diff_var / 2)
    readout_noise_e = readout_noise_adu * gain

    print(f"Readout noise = {readout_noise_e:.3f} e⁻")
  
    return readout_noise_e
    
if __name__ == "__main__":
    readout = calculate_readout_noise()