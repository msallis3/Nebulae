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

    plt.imshow(reduced_science, cmap = 'inferno', origin = 'lower', aspect = 'auto')
    plt.colorbar()
    plt.savefig(Path("Twin_Images") / 'Twin_reduced_science.png')
    plt.tight_layout()
    plt.close()
    

    #Saving
    primary = fits.PrimaryHDU(data=reduced_science, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(output_file, overwrite=True)

    return 



def do_aperture_photometry(positions, radii):

    output_file = Path("Twin_Images") / 'Twin_aperture_photometry.fits'
    input_dir = Path("Twin_Images")

    
    
    #Opening the data and turning it into flaot 32
    for f in sorted(input_dir.glob('Twin_Images/twin_science_reduced.fits')):
        with fits.open(f) as image:
            image_data = image[0].data.astype('f4')

    rows = []

    #Doing a loop over the star
    for p in positions:
        #Creating dictionary with stars position
        row = {'xcenter': p[0], 'ycenter': p[1]}

        #Looping over the radii for each star
        for r in radii:
            
            # Doing the aperture and annulus stuff for one star
            aperture = CircularAperture(p, r=r)
            annulus = CircularAnnulus(p, r_in=sky_radius_in, r_out=sky_radius_in + sky_annulus_width)

            # Calculating the aperture and annulus photometry
            phot_ap = aperture_photometry(image_data, aperture)
            stats = ApertureStats(image_data, annulus, sigma_clip=None)

            # finding area of circle and subtracting the sky background to find total flux
            aperture_area = aperture.area
            background = stats.median * aperture_area
            flux = phot_ap['aperture_sum'][0] - background

            #Storing results in dictionary with star position and flux 
            row[f'flux_r{int(r)}'] = flux
            plt.plot(flux, label=f'Radius {r}')
            
        plt.imshow(image_data, cmap = 'inferno', origin = 'lower', aspect = 'auto')
        plt.colorbar()
        plt.savefig(Path("Twin_Images") / 'Twin_photometry.png')
        plt.tight_layout()
        plt.close()

        #Putting the data into emtpy list
        rows.append(row)

    # Creating astropy table from dictionary data and adding radii/sky as metadata stuff
    table = Table(rows)
    table.meta['radii'] = radii
    table.meta['sky_radius'] = sky_radius_in

    plt.imshow(image_data, cmap = 'inferno', origin = 'lower', aspect = 'auto')
    plt.colorbar()
    plt.savefig(Path("Twin_Images") / 'Twin_photometry.png')
    plt.tight_layout()
    plt.close()

    table.write(output_file, format='fits', overwrite=True)

    
    return 
#if __name__ == "__main__":
 #   photometry = do_aperture_photometry(positions, radii)
  #  print("It has printed")

def plot_radial_profile(aperture_photometry_data, output_filename="radial_profile.png"):

    #Calling radii from aperture data as a array        
    radii = np.array(aperture_photometry_data.meta['radii'], dtype = float)

    #Getting the sky radius, also from aperture data
    sky_radius = aperture_photometry_data.meta['sky_radius']

    #taking first flux data for each r 
    fluxes = [aperture_photometry_data[f'flux_r{int(r)}'][0] for r in radii]

    #Plotting radii and fluxes
    plt.figure()
    plt.plot(radii, fluxes, marker='o', label='Target')

    #Sky radius position on graph 
    plt.axvline(sky_radius, color='gray', linestyle='--', label='Sky radius')

    #Making it look nice
    plt.xlabel('Radius')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()

    #Saving everything
    plt.savefig(output_filename)
    plt.close()

def calculate_gain(files):

    #Creating empty list to store data in
    flats = []

    #Opening files data
    for f in files:
        with fits.open(f) as file: 

            #Trimming data
            trim = file[0].data[1600:2000, 1300:1700]

            #Putting it into empty list
            flats.append(trim)

    #Unpacking flats
    flats1, flats2 = flats

    #Getting the difference of the two flats
    flat_diff = flats1 - flats2

    #Calculating variance 
    flat_var = np.var(flat_diff)

    #Getting average between the two
    mean = 0.5 * np.mean(flats1 + flats2)

    #Getting the gain with formula 
    gain = 2 * mean / flat_var

    return gain


def calculate_readout_noise(files, gain):

    #Creating another empty list
    file_data = []

    #Opening files and trimming then putting into list
    for f in files:
        with fits.open(f) as file: 
            trim = file[0].data[1000:-1000, 1000:-1000]
            file_data.append(trim)

    #Unpacking files
    bias1, bias2 = file_data
            
    # Calculate the variance of the difference between the two images
    bias_diff = bias1 - bias2
    bias_diff_var = np.var(bias_diff)

    # Calculate the readout noise
    readout_noise_adu = np.sqrt(bias_diff_var / 2)
    readout_noise_e = readout_noise_adu * gain

    return readout_noise_e


def run_reduction(data_dir):
    
    data = Path(data_dir)

    #Getting bias data
    median_bias_path = data / "median_bias.fits"
    bias_files = list(data.glob("Bias*.fit"))
    create_median_bias(bias_files, median_bias_path)
    median_bias = fits.getdata(median_bias_path).astype('f4')

    #Getting dark data
    median_dark_path = data / "median_dark.fits"
    dark_files = list(data.glob("Dark*.fit"))
    create_median_dark(dark_files, median_bias_path, median_dark_path)
    median_dark = fits.getdata(median_dark_path).astype('f4')

    #Getting flat data
    median_flat_path = data / "median_flat.fits"
    flat_files = list(data.glob("AutoFlat*.fit"))
    create_median_flat(flat_files, median_bias_path, median_flat_path, dark_filename=median_dark_path)
    median_flat = fits.getdata(median_flat_path).astype('f4')

    #getting science data
    science_files = sorted(data.glob("kelt-16-b-S001-R001-C*-r.fit"))

    #Creating empty list to put stuff in later
    science = []

    #Opening science files
    for s in science_files:
        
       #Making a path so it works correctly
        output_filename = data / (s.stem + "_reduced.fits")


        #Doing the actual reductions of image, something wrong here 
        reduced_path = reduce_science_frame(s, median_bias_path, median_dark_path, median_flat_path,reduced_science_filename = output_filename)

        
        print("Type of reduced_path:", type(reduced_path))

        #putting it all back into llist
        science.append(str(reduced_path))

    #Trying to open it as an image, I am not sure it is actually passing as an image
    if science:
        image = science[0]
        with fits.open(image) as hdul:
            image_data = hdul[0].data.astype('f4')

        # Doing the stuff i did in one of the functions and pulling from prof examples
        #Getting stars
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
        sources = daofind(image_data - median)

        #Getting positions
        positions = [(src['xcentroid'], src['ycentroid']) for src in sources]

        # giving it data, idk how to pull this like i did in the functions, I am so tired
        aperture_radius = 4
        sky_radius_in = 6
        sky_annulus_width = 2

        #This will not work, I am passing something as a string when I am not suppposed to:(
        #Trying to run photometry and save it
        phot_table = do_aperture_photometry(image_data, positions, aperture_radius, sky_radius_in, sky_annulus_width, plot_path=data / "photometry_plot.png")

        print(phot_table)

        #Trying to save it
        phot_table.write(data /"photometry_results.csv", format="csv", overwrite=True)


    return 


