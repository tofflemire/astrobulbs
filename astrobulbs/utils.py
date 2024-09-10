import pickle as pkl
import numpy as np
import os as os
import copy as copy
import glob as glob
import sys

import saphires as saph

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import patches

from scipy.optimize import curve_fit

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.visualization import MinMaxInterval, ZScaleInterval, ImageNormalize, SqrtStretch, simple_norm
from astropy.time import Time
from astropy import units as u
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData, CCDData
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

from astroquery.gaia import Gaia
from astroquery.astrometry_net import AstrometryNet

from ccdproc import Combiner

from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.psf import EPSFBuilder
from photutils.background import MMMBackground
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import IterativelySubtractedPSFPhotometry, DAOGroup, extract_stars
from photutils.aperture import ApertureStats
from photutils.utils import calc_total_error


def import_images(im_list, p):
    '''
    A function that imports the data from an image file, following a given
    path to find the image file

        Paramters
        ---------
        im_list: list
            List containing the names of the image files
        p: string
            The pathway the script should follow to find the image
            files on the computer

        Returns
        -------
        im_data: list
            A list of the data arrays containing the pixel data
            of images
        in_headers: list
            A list of all the image headers
    '''
    im_data = []
    im_headers = []
    for i in im_list:
        x = str(i)
        path = p + x
        hdu = fits.open(path)
        data = hdu[1].data
        header = hdu[1].header
        im_data.append(data)
        im_headers.append(header)

    return im_data, im_headers

def find_fwhm(image_in, size=30, default=7):
    '''
    Fits a 2D gaussian surface to the brightest, non-saturated star
    on an image

        Parameters
        ----------
        image: array-like
            raw pixel data from the image
        size: integer
            radius of the cutout around the star

        Returns
        -------
        popt: list
            list of all the best fit values of the gaussians parameters:
            x0, y0, sig_x, sig_y, Amplitude, offset (background estimate)
    '''
    image = copy.deepcopy(image_in)
    mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=2.0)
    search_image = image[100:-100,100:-100]
    max_peak = np.max(search_image)
    count = 0
    while max_peak >= 0:
        count += 1
        rs, cs = np.where(search_image==max_peak)[0][0], np.where(search_image==max_peak)[1][0]
        r = rs+100
        c = cs+100
        if max_peak < 50000:
            star = image[r-size:r+size+1,c-size:c+size+1]
            x = np.arange(2*size+1)
            y = np.arange(2*size+1)
            X, Y = np.meshgrid(x, y)

            im_med = np.median(star[:15,:15])
            im_std = np.std(star[:15,:15])

            n_pix_good = np.sum(star > im_med+5*im_std)

            if n_pix_good > 50:

                def gaussian(M, x0, y0, sig_x, sig_y, A, off):
                    x, y = M

                    prof = A * np.exp(-((x-x0)**2)/(2*sig_x**2)-((y-y0)**2)/(2*sig_y**2)) + off

                    return prof

                def gaussian_PA(M, x0, y0, sig_x, sig_y, A, theta, off):
                    x, y = M

                    x_rotated = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
                    y_rotated = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
                    prof = A * np.exp(-(x_rotated**2 / (2 * sig_x**2) + y_rotated**2 / (2 * sig_y**2))) + off

                    return prof
    
                xdata = np.vstack((X.ravel(), Y.ravel()))
                ydata = star.ravel()
                
                p = [size, size, 3, 3, 10000, 0, median_val]
                #p = [size, size, 3, 3, 10000, median_val]
                bounds = [(0,0,1,1,100,-np.pi,-10000),
                          (size*2,size*2,size,size,100000,np.pi,10000)]

                try:
                    popt, pcov = curve_fit(f=gaussian_PA, xdata=xdata, ydata=ydata, p0=p, bounds=bounds)
                    x_fwhm = popt[2]*gaussian_sigma_to_fwhm
                    y_fwhm = popt[3]*gaussian_sigma_to_fwhm
                    theta = popt[5]

                    if x_fwhm < y_fwhm:
                        print('major axis flip, check behavior')
                        print(x_fwhm,y_fwhm,theta)
                        fwhm = y_fwhm
                        theta = np.pi/2.0 + theta
                        ratio = x_fwhm / y_fwhm


                    else:
                        fwhm = x_fwhm
                        ratio = y_fwhm / x_fwhm

                    im_sig = fwhm/gaussian_sigma_to_fwhm
                    
                except:
                    fwhm = 0
    
                if ((fwhm > 4)&(fwhm < 30)):
                    break
                else:
                    image[r-size:r+size+1,c-size:c+size+1] = 0
                    search_image = image[100:-100,100:-100]
                    max_peak = np.max(search_image)

            else:
                image[r-size:r+size+1,c-size:c+size+1] = 0
                search_image = image[100:-100,100:-100]
                max_peak = np.max(search_image)
        else:
            image[r-size:r+size+1,c-size:c+size+1] = 0
            search_image = image[100:-100,100:-100]
            max_peak = np.max(search_image)

        if count > 100:
            fwhm = default
            im_sig = default/gaussian_sigma_to_fwhm
            ratio = 1
            theta = 0
            break

        #if max_peak < 1000:
        #    fwhm = 0
        #    im_sig = 0
        #    break

    return fwhm, im_sig, star, r, c, theta, ratio

def find_stars(image, sigma, ratio, theta, peak=100000,threshold = 5,mask = None):
    '''
    Searches data from an image to find objects above a given brightness
    threshold based off parameters of the ccd chip

        Parameters
        ----------
        image: array-like
            Array containing the intensity of light for each pixel
            on the ccd chip
        sigma: float
            sets the size tolerance for detected objects. Usually
            5.0, more than 5 is statistically unreasonable
        peak: int
            The max number of counts the chip can handle before the
            image starts to smear. Usual ccd can handle 100,000 counts

        Returns
        -------
        stars: table
            A table containing all the found stars and their parameters:
            id, xcentroid, ycentroid, sharpness, roundness, npix, sky,
            peak, flux, mag
    '''
    sigma_psf = sigma
    mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=3.0)

    #print(sigma_psf*gaussian_sigma_to_fwhm,median_val+threshold*std_val)

    daofind = DAOStarFinder(fwhm=sigma_psf*gaussian_sigma_to_fwhm, 
                            ratio = ratio,
                            theta = theta,
    						threshold=median_val+threshold*std_val,
                            sky=0.0, peakmax=peak, exclude_border=True)
    stars = daofind(image, mask = mask)
    return stars

def calculate_shift(stars1, stars2, match_thresh=50):
    '''
    Calculates the necessary shift of one image in order to be aligned
    with a second image

        Parameters
        ----------
        stars1: table
            The table returned from using find_stars on an image
        stars2: table
            Same as stars1, for a different image

        Returns
        -------
        diff: table
            Table containing the x, y, and total offset of each star object
            found between two images
    '''
    diff = np.zeros([stars1['xcentroid'].size, 3])*np.nan
    for i in range(stars1['xcentroid'].size):
        dx = stars1['xcentroid'][i] - stars2['xcentroid']
        dy = stars1['ycentroid'][i] - stars2['ycentroid']
        distances = np.abs(np.sqrt((dx)**2 + (dy)**2))
        match = (distances == np.min(distances))
        if distances[match] < match_thresh:
            diff[i, 0] = distances[match][0]
            diff[i, 1] = dx[match][0]
            diff[i, 2] = dy[match][0]

    return diff

def roll_image(image, diff, threshold=0.5):
    '''
    Averages the x and y offset of objects on 2 images to the nearest
    integer, and then rolls the image by that number of pixels along each
    axis. Good for aligning two images

        Parameters
        ----------
        image: array-like
            Array containing the intensity of light for each pixel
            on the ccd chip
        diff: table
            Table containing the x, y, and total offset of each star object
            found between two images
        threshold: float
            The minimum pixel offset between images to allow shifting,
            usually 0.5 pixels

        Returns
        -------
        image_shift: array-like
            The "rolled" version of the same image, now aligned to another
            reference image
    '''
    offset = np.nanmedian(diff[:, 0])
    xshift = np.nanmedian(diff[:, 1])
    yshift = np.nanmedian(diff[:, 2])
    xshift_int = int(np.round(xshift, 0))
    yshift_int = int(np.round(yshift, 0))
    if np.max(np.abs([xshift_int,yshift_int])) > threshold:
        image_shift = np.roll(image, (yshift_int, xshift_int), axis = (0, 1))

        #print('Shifts:',xshift_int,yshift_int)

        return image_shift, xshift_int,yshift_int
    else:
        return image, xshift_int,yshift_int

def median_combiner(images):
    '''
    Function that takes the median of multiple images containing the
    same stars objects

        Parameters
        ----------
        images: list
            A list of the data arrays containing the pixel data
            of images

        Returns
        -------
        median_image: array-like
            Array containing the median intensity of light for each
            pixel for a set of images
    '''
    ccd_image_list = []

    for image in images:
        ccd_image = CCDData(image, unit=u.adu)
        ccd_image_list.append(ccd_image)

    c = Combiner(ccd_image_list)
    c.sigma_clipping(func = np.ma.median)
    median_image = c.median_combine()
    median_image = np.asarray(median_image)

    return median_image

def image_combiner(im_data, im_sig):
    '''
    Returns a median combination of a list of images that has been 
    aligned to match star locations.

        Parameters
        ----------
        im_data: list
            contains all the image data from the image set
        im_sig: float
            an image customized size parameter for searching an
            image for stars

        Returns
        -------
        median_image: array-like
    '''
    stars = []
    for i in im_data:
        s = find_stars(image=i, sigma=im_sig, peak=100000)
        stars.append(s)
    
        if s is None:
            median_image = None
            return median_image
    

    diffs = []
    for s in range(len(stars)):
            diff = calculate_shift(stars1=stars[0], stars2=stars[s])
            diffs.append(diff)
    images = []
    xshift_ints = []
    yshift_ints = []

    #this centroid shifting thing is hit or miss, probably more miss. 
    for i in range(len(im_data)):
        image_shift,xshift_int,yshift_int = roll_image(image=im_data[i], diff=diffs[i], threshold=0.5)
        xshift_ints.append(xshift_int)
        yshift_ints.append(yshift_int)
        images.append(image_shift)
    median_image = median_combiner(images=images)

    return median_image, xshift_ints, yshift_ints

def image_mask(image, sources, fwhm, bkg, bkg_std):
    '''
    Masking routine that rejects stars too close to the edge of the
    image, too close to each other, and the 5 brightest and 5 dimmest
    stars in the image

        Parameters
        ----------
        image: array-like
            raw pixel data from the image
        sources: Table
            contains all the data aquired from the star searching routine
        fwhm: float
            used for scaling the mask based on how focused the image is

        Returns
        -------
        stars_tbl: Table
            condensed form of the sources table, excluding all the masked
            stars. columns: xcentroid, ycentroid, flux, peak, id
    '''
    size = 100
    hsize = (size - 1) / 2
    x = sources['xcentroid']
    y = sources['ycentroid']
    flux = sources['flux']
    i = sources['id']
    p = sources['peak']
    mask = ((x > hsize) & (x < (image.shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (image.shape[0] - 1 - hsize)))
    stars_tbl = Table()
    stars_tbl['x'] = x[mask]
    stars_tbl['y'] = y[mask]
    stars_tbl['flux'] = flux[mask]
    stars_tbl['id'] = i[mask]
    stars_tbl['peak'] = p[mask]
    stars_tbl.sort('flux', reverse=True)

    d = []
    idxi = 0
    for i in stars_tbl['id']:
        idxj = 0
        for j in stars_tbl['id']:
            if i != j:
                threshold = 5*fwhm
                dx = stars_tbl['x'][idxi] - stars_tbl['x'][idxj]
                dy = stars_tbl['y'][idxi] - stars_tbl['y'][idxj]
                distance = np.abs(np.sqrt((dx)**2 + (dy)**2))
                if distance <= threshold:
                    d.append(idxi)
            idxj = idxj+1
        idxi = idxi + 1

    idxp = 0
    min_peak = bkg + 10 * bkg_std
    for i in stars_tbl['peak']:
        if i <= min_peak:
            d.append(idxp)
        idxp += 1
    stars_tbl.remove_rows(d)
    stars_tbl.sort('flux', reverse=True)

    #remove the two brightest and dimmest stars, trying to avoid saturation
    if len(stars_tbl) > 10:
        stars_tbl.remove_rows([-2,-1,0,1])

    return stars_tbl

def bkg_sub(image, stars_tbl, fwhm):
    '''
    Local background subtraction routine for stars on an image

        Parameters
        ----------
        image: array-like
            raw pixel data of the image
        stars_tbl: Table
            contains positional and flux data for all the stars
        fwhm: float
            used for scaling the area to be background subtracted
            based on how focused the image is

        Returns
        -------
        image_lbs: array-like
            a copy of the original image, with regions around each star
            containing no background flux
    '''
    image_lbs = copy.deepcopy(image)
    for s in stars_tbl['x','y']:
        position = [s[0],s[1]]
        aperture = CircularAperture(position, r=20)
        annulus = CircularAnnulus(position, r_in=20, r_out=30)
        annulus_mask = annulus.to_mask(method='center')
        annulus_data = annulus_mask.multiply(image_lbs)
        annulus_data_1d = annulus_data[annulus_mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median = median_sigclip
        pos_pix = [int(np.round(position[0], 0)), int(np.round(position[1], 0))]
        size = 5*fwhm
        for r in range(len(image_lbs)):
            if (r > pos_pix[1]-(size/2) and r < pos_pix[1]+(size/2)):
                for c in range(len(image_lbs[r])):
                    if (c > pos_pix[0]-(size/2) and c < pos_pix[0]+(size/2)):
                        image_lbs[r][c] -= bkg_median

    return image_lbs

def build_psf(image, stars_tbl, fwhm):
    '''
    Constructs a poins spread function (psf) from a sample of stars
    on an image

        Parameters
        ----------
        image: array-like
            raw pixel data of the image
        stars_tbl: Table
            contains positional and flux data for all the stars
        fwhm: float
            used for scaling the size of the star cutouts based on
            how focused the image is

        Returns
        -------
        epsf: EPSFModel
            the effective psf constructed form the stars
        stars: EPSFStars
            the star cutouts used to build the psf
        fitted_stars: EPSFStars
            the original stars, with updated centers and fluxes derived
            from fitting the output psf
    '''
    nddata = NDData(data = image)

    size = int(5*fwhm)

    if (size % 2) == 0:
    	size = size + 1

    stars = extract_stars(nddata, stars_tbl, size = size)
    epsf_builder = EPSFBuilder(oversampling=2, maxiters=10, progress_bar=False, smoothing_kernel='quadratic')
    epsf, fitted_stars = epsf_builder(stars)

    return epsf, stars, fitted_stars

def do_photometry(image, epsf, fwhm, mask = None):
    '''
    Iterative photometry routine using a point spread function (psf) model

        Parameters
        ----------
        image: array-like
            raw pixel data from the image
        epsf: EPSFModel
            the psf model for finding stars on the image
        fwhm: float
            used for scaling data collection region around each star based
            on how focused the image is

        Returns
        -------
        results: Table
            contains all the photometry data: x_0, x_fit, y_0, y_fit, flux_0,
            flux_fit, id,group_id, flux_unc, x_0_unc, y_0_unc, iter_detected
        photometry:
            the iterative search function for performing photometry
    '''
    mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=2.0)
    daofind = DAOStarFinder(fwhm=fwhm, 
    						threshold=median_val+7*std_val, 
    						sky=median_val, 
    						peakmax=100000, 
    						exclude_border=True,
    						sharphi=0.6)
    daogroup = DAOGroup(5*fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    def round_to_odd(f):
        return np.ceil(f) // 2 * 2 + 1
    size = 5*fwhm
    fitshape = int(round_to_odd(size))
    photometry = IterativelySubtractedPSFPhotometry(finder=daofind, group_maker=daogroup, bkg_estimator=mmm_bkg,
                                                    psf_model=epsf, fitter=fitter, niters=5, fitshape=fitshape,
                                                    aperture_radius=(size-1)/2)
    results = photometry(image, mask = mask)

    return results, photometry

def get_residuals(results, photometry, fwhm, image):
    '''
    Generates residual image cutouts from photometry results

        Parameters
        ----------
        results: Table
            contains all the photometry data: x_0, x_fit, y_0, y_fit, flux_0,
            flux_fit, id,group_id, flux_unc, x_0_unc, y_0_unc, iter_detected
        photometry:
            the iterative search function for performing photometry

        Results
        -------
        results_tbl: Table
            condensed table of the photometry results, with just the positional
            and flux data
        residual_stars: EPSFStars
            cutouts of the residuals of the stars left after photometry is completed
    '''
    results_tbl = Table()
    results_tbl['x'] = results['x_fit']
    results_tbl['y'] = results['y_fit']
    results_tbl['flux'] = results['flux_fit']
    results_tbl.sort('flux', reverse=True)
    ndresidual = NDData(data=photometry.get_residual_image())
    nddata = NDData(data=image)

    size = int(5*fwhm)
    if (size % 2) == 0:
    	size = size + 1
    
    
    final_stars = extract_stars(nddata, results_tbl, size=size)
    residual_stars = extract_stars(ndresidual, results_tbl, size=size)

    return results_tbl, final_stars, residual_stars

def get_wcs(results_tbl):
    '''
    Queries the website astrometry.net with image data, returning a world coordinate
    system (wcs) solution, along with a header containing this solution

        Parameters
        ----------
        results_tbl: Table
            contains positional and flux data for all stars found in the photometry
            routine

        Results
        -------
        sky: Table
            contains all locations for the stars in results_tbl in RA and DEC
            instead of pixels
        wcs_header: Header
            an image header with the RA and DEC included
    '''
    ast = AstrometryNet()
    ast.api_key = 'kxkjdlaxzzfubpws'
    try_again = True
    submission_id = None
    image_width = 2042
    image_height = 3054
    while try_again:
        try:
            if not submission_id:
                wcs_header = ast.solve_from_source_list(results_tbl['xcenter'][:30].value, 
                										results_tbl['ycenter'][:30].value,
                                                        image_width, image_height, submission_id=submission_id,
                                                        solve_timeout=600)
            else:
                wcs_header = ast.monitor_submission(submission_id, solve_timeout=600)
        except TimeoutError as e:
            submission_id = e.args[1]
        else:
            try_again = False

    if wcs_header:
        w = WCS(wcs_header)
        sky = w.pixel_to_world(results_tbl['xcenter'], results_tbl['ycenter'])
        return sky, wcs_header, w
    else:
        return None, wcs_header, None

def write_csv(name, im_name, bjd, filt, airmass, results, sky):
    f = open(name, 'w')
    f.write('NAME, ID, BJD, FLUX, FLUX ERROR, MAG, MAG ERROR, FILTER, X POSITION, Y POSITION, AIRMASS, RA, DEC\n')
    for i in range(sky.size):
        if results['flux_fit'][i] > 0:
            star_id = results['id'][i]
            flux = results['flux_fit'][i]
            fluxerr = results['flux_unc'][i]
            mag = -2.5*np.log10(flux)
            magerr = (1.08574*fluxerr)/(flux)
            x_pos = results['x_fit'][i]
            y_pos = results['y_fit'][i]
            ra = sky[i].ra.degree
            dec = sky[i].dec.degree
            f.write(im_name+','+str(i)+','+str(bjd)+','+str(flux)+','+str(fluxerr)+','+str(mag)+','+str(magerr)
                    +','+filt+','+str(x_pos)+','+str(y_pos)+','+str(airmass)+','+str(ra)+','+str(dec)+'\n')
    f.close()

def write_pdf(name, images, model=None, final_stars=None, residual_stars=None, fluxes=None, plot_res=None):
    pp = PdfPages(name)
    for i in range(len(images)):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        norm = ImageNormalize(images[i], interval=ZScaleInterval(), stretch=SqrtStretch())
        im = ax.imshow(images[i], norm=norm)
        plt.colorbar(im)
        plt.tight_layout()
        pp.savefig()
        plt.close()
    if model is not None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        psf = ax.imshow(model)
        plt.colorbar(psf)
        ax.set_title('PSF Model')
        plt.tight_layout()
        pp.savefig()
        plt.close()
    if final_stars is not None:
        if plot_res == 'y':
            nrows = 50
            ncols = 2
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 500), squeeze=True)
            ax = ax.ravel()
            index = 0
            for i in range(0, nrows*ncols, 2):
                norm = simple_norm(final_stars[index],'log')
                norm2 = simple_norm(residual_stars[index], 'linear')
                im = ax[i].imshow(final_stars[index], norm=norm, origin='lower', cmap='viridis', interpolation='none')
                fig.colorbar(im, ax = ax[i])
                ax[i].set_title(str(fluxes[index]))
                im_r = ax[i+1].imshow(residual_stars[index], norm=norm2, origin='lower', cmap='viridis', interpolation='none')
                fig.colorbar(im_r, ax = ax[i+1])
                index = index + 1
            plt.tight_layout()
            pp.savefig()
            plt.close()
    pp.close()

def aperture_phot(image, fwhm, ratio, theta, threshold = 5):

    sigma = fwhm / gaussian_sigma_to_fwhm

    #find the stars
    stars = find_stars(image, sigma, ratio, theta, threshold = threshold)

    #array of the pixel positions
    positions = np.array([stars['xcentroid'].value,stars['ycentroid'].value]).T.reshape(stars['xcentroid'].value.size,2)

    #Create apertures and annuli
    aperture = CircularAperture(positions, r=int(fwhm*3))
    annulus = CircularAnnulus(positions, r_in=int(fwhm*4), r_out=int(fwhm*5))

    #Defines a sigma clipping function, i.e., not a value
    sigclip = SigmaClip(sigma=3.0, maxiters=10)

    #Aperture photometry including background
    aperstats = ApertureStats(image, aperture, sigma_clip=None)

    #Background statistics
    bkgstats = ApertureStats(image, annulus, sigma_clip=sigclip)
    total_bkg = bkgstats.median * aperstats.sum_aper_area.value

    #Aperture sum with the background subtracted
    apersum_bkgsub = aperstats.sum - total_bkg

    #Error on the background subtracted aperture sum - here the gain is 1 so it is exlcuded
    apersum_bkgsub_err = np.sqrt(bkgstats.std**2 + apersum_bkgsub)

    phot_table = Table()
    phot_table['xcenter'] = stars['xcentroid'].value
    phot_table['ycenter'] = stars['ycentroid'].value
    phot_table['aperture_sum_bkgsub'] = apersum_bkgsub
    phot_table['aperture_sum_bkgsub_err'] = apersum_bkgsub_err

    phot_table.sort('aperture_sum_bkgsub', reverse=True)

    phot_table['id'] = np.arange(stars['xcentroid'].value.size)


    #bkg_median = ApertureStats(image, annulus, sigma_clip=sigclip).median
    #bkg_std = ApertureStats(image, annulus, sigma_clip=sigclip).std

    #mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=2.0)

    #error = calc_total_error(image,median_val,1)
    
    #aper_stats_bkgsub = ApertureStats(data, aperture, local_bkg=bkg_stats.median)

    #phot_table = aperture_photometry(image, aperture, error=error)

    #total_bkg = bkg_median*aperture.area_overlap(image)
    #total_bkg_err = bkg_std*aperture.area_overlap(image)
    #phot_bkgsub = phot_table['aperture_sum'] - total_bkg

    #phot_table['total_bkg'] = total_bkg
    #phot_table['total_bkg_err'] = total_bkg_err
    #phot_table['aperture_sum_bkgsub'] = phot_bkgsub
    #phot_table['aperture_sum_bkgsub_err'] = np.sqrt(phot_table['aperture_sum_err']**2 + total_bkg_err**2)


    return phot_table

def ap_phot_cutouts(image, phot_table, fwhm):

	nddata = NDData(data=image)

	results_tbl = Table()
	results_tbl['x'] = phot_table['xcenter']
	results_tbl['y'] = phot_table['ycenter']

	size = int(6*fwhm)
	if (size % 2) == 0:
		size = size + 1

	cutouts = extract_stars(nddata, results_tbl, size=size)

	return cutouts, size

def write_csv_apphot(name, im_name, bjd, filt, airmass, results, sky):
    f = open(name, 'w')
    f.write('#NAME, ID, BJD, FLUX, FLUX ERROR, MAG, MAG ERROR, FILTER, X POSITION, Y POSITION, AIRMASS, RA, DEC\n')
    for i in range(sky.size):
        if results['aperture_sum_bkgsub'][i] > 0:
            star_id = results['id'][i]
            flux = results['aperture_sum_bkgsub'][i]
            fluxerr = results['aperture_sum_bkgsub_err'][i]
            mag = -2.5*np.log10(flux)
            magerr = (1.08574*fluxerr)/(flux)
            x_pos = results['xcenter'][i]#.value
            y_pos = results['ycenter'][i]#.value
            ra = sky[i].ra.degree
            dec = sky[i].dec.degree
            f.write(im_name+','+str(i)+','+str(bjd)+','+str(flux)+','+str(fluxerr)+','+str(mag)+','+str(magerr)
                    +','+filt+','+str(x_pos)+','+str(y_pos)+','+str(airmass)+','+str(ra)+','+str(dec)+'\n')
    f.close()

def organize_raw_images(target,filt_name,n_set = 3):

    f_names = glob.glob('*.fz')

    filt = np.zeros(len(f_names),dtype="U100")
    datestr = np.zeros(len(f_names),dtype="U100")
    date = np.zeros(len(f_names))
    dateobs = np.zeros(len(f_names),dtype="U100")
    telid = np.zeros(len(f_names),dtype="U100")
    molnum = np.zeros(len(f_names),dtype=int)
    molfrnum = np.zeros(len(f_names),dtype=int) 
    blkuid = np.zeros(len(f_names),dtype=int) 
    exptime = np.zeros(len(f_names))

    obsid = np.zeros(len(f_names),dtype=int)
    imageid = np.zeros(len(f_names),dtype=int)
    nightid = np.zeros(len(f_names),dtype=int)

    for i in range(len(f_names)):
        hdu = fits.open(f_names[i])
        
        filt[i] = hdu[1].header['FILTER']
        datestr[i] = f_names[i][14:22]
        date[i] = hdu[1].header['MJD-OBS']
        dateobs[i] = hdu[1].header['DATE-OBS']
        telid[i] = hdu[1].header['TELESCOP']
        molnum[i] = hdu[1].header['MOLNUM']
        molfrnum[i] = hdu[1].header['MOLFRNUM']
        blkuid[i] = hdu[1].header['BLKUID']
        exptime[i] = hdu[1].header['EXPTIME']

    sequence_start = np.zeros(len(f_names)) 

    blocks = np.unique(blkuid)

    for i in range(len(f_names)):
        match = ((blkuid == blkuid[i]) & (molfrnum == 1))
        sequence_start[i] = date[match][0]

    f_names = np.array(f_names)[np.lexsort((date,sequence_start))]
    filt = filt[np.lexsort((date,sequence_start))]
    datestr = datestr[np.lexsort((date,sequence_start))]
    dateobs = dateobs[np.lexsort((date,sequence_start))]
    telid = telid[np.lexsort((date,sequence_start))]
    molnum = molnum[np.lexsort((date,sequence_start))]
    molfrnum = molfrnum[np.lexsort((date,sequence_start))]
    blkuid = blkuid[np.lexsort((date,sequence_start))]
    exptime = exptime[np.lexsort((date,sequence_start))]

    date = date[np.lexsort((date,sequence_start))]

    nightnum = 0
    idnum = 0
    #imagenum = 0
    
    blocks = np.unique(blkuid) #now sorted by the start time of the sequence

    for i in range(blocks.size):
        match = (blkuid == blocks[i])

        datestr_i = datestr[match][0]

        if np.sum(match) < n_set:
            print('Block '+str(blocks[i])+' is incomplete')

        if np.sum(match) > n_set:
            print('Block '+str(blocks[i])+' has too many images')

        obsid[match] = idnum

        imageid[match] = molfrnum[match]-1

        nightid[match] = nightnum

        idnum += 1

        if i < blocks.size-1:
            datestr_n = datestr[blkuid == blocks[i+1]][0]
        
            if datestr_i != datestr_n:
                nightnum = 0
            else:
                nightnum += 1

    f = open(target+'_epochs_'+filt_name+'.csv','w')
    
    for i in range(f_names.size):
        f.write(f_names[i]+','+datestr[i]+','+str(nightid[i])+','+str(imageid[i])+','+str(obsid[i])+','+str(blkuid[i])+','+dateobs[i]+','+str(exptime[i])+','+telid[i]+'\n')

        print(f_names[i],datestr[i],nightid[i],imageid[i],molfrnum[i]-1,obsid[i],)

    f.close()


#    for i in range(obsid.size):
#        
#        obsid[i] = int(idnum)
#        imageid[i] = int(imagenum)
#        nightid[i] = int(nightnum)
#
#        if filt[i] == filt_name:
#            
#            print(f_names[i],datestr[i],nightid[i],imageid[i],molfrnum[i]-1,obsid[i],)
#
#            if imageid[i] != molfrnum[i]-1:
#                print('Something is wrong with the matching')
#        
#            f.write(f_names[i]+','+datestr[i]+','+str(nightid[i])+','+str(imageid[i])+','+str(obsid[i])+','+str(blkuid[i])+','+str(dateobs[i])+','+str(exptime[i])+','+str(telid[i])+'\n')
#    
#            if i < obsid.size-1:
#                if datestr[i+1] != datestr[i]:
#                    nightnum = 0
#                    idnum +=1
#                    imagenum = 0
#
#                if ((datestr[i+1] == datestr[i]) & (imagenum < n_set)):
#                    imagenum += 1
#
#                if ((datestr[i+1] == datestr[i]) & (imagenum > n_set-1)):
#                    imagenum = 0
#                    nightnum += 1
#                    idnum += 1

    f.close()

    return f_names,filt,obsid,imageid,datestr,nightid

def photometry_wrap(target,tra,tdec,names,filt,eid,iid,filt_i,eid_i,nightid_i,datestr_i):

    print('')

    nightid_out = nightid_i[(eid == eid_i)][0]
    datestr_out = datestr_i[(eid == eid_i)][0]

    name_out = target+'_'+filt_i+'_'+datestr_out+'_'+str(nightid_out)+'_'+str(int(eid_i))

    if os.path.exists('./'+name_out+'.csv'):
        print(name_out+' is already finished, skipping.')

    else:
        pp = PdfPages(name_out+'.pdf')

        #stars_tbl = Table()


        #SELECT RELEVANT IMAGES AND INFORMATION
        im_1 = names[(filt == filt_i)&(eid == eid_i)&(iid == 0)][0]
        hdu_1 = fits.open(im_1)
        im_1_data = hdu_1[1].data
        im_1_header = hdu_1[1].header
        
        tele = ''

        if hdu_1[1].header['SITEID'] == 'tfn':
            tele = 'lco-tfn'

        if hdu_1[1].header['SITEID'] == 'ogg':
            tele = 'lco-ogg'

        if hdu_1[1].header['SITEID'] == 'lsc':
            tele = 'lsc'

        if hdu_1[1].header['SITEID'] == 'cpt':
            tele = 'cpt'
        
        if hdu_1[1].header['SITEID'] == 'coj':
            tele = 'coj'

        if hdu_1[1].header['SITEID'] == 'elp':
            tele = 'elp'

        if hdu_1[1].header['SITEID'] == 'tlv':
            tele = 'tlv'

        im_data = []
        
        imset = names[(filt == filt_i)&(eid == eid_i)]
        print(imset)
        
        airmass = np.zeros(imset.size)
        jd_middle = np.zeros(imset.size)
        fwhm_lco = np.empty(0)

        stars = []
        
        for j in range(imset.size):
            hdu = fits.open(imset[j])
        
            image = hdu[1].data

            im_data.append(image)
        
            airmass[j] = hdu[1].header['AIRMASS']
            jd = Time(hdu[1].header['DATE-OBS'],format='isot').jd
            jd_middle[j] = jd + (hdu[1].header['EXPTIME']/2.0)/3600.0/24.0

            norm = ImageNormalize(im_data[j], interval=ZScaleInterval(), stretch=SqrtStretch())

            fwhm_lco = np.append(fwhm_lco,hdu[2].data['fwhm'])

            if j == 0:
                im_1_fwhm_f,im_1_sig_f,im_1_star,r_1,c_1,im_1_theta,im_1_ratio = find_fwhm(image)

                #trying the LCO fwhm measurement
                im_1_fwhm_l = np.nanmedian(fwhm_lco)
                im_1_sig_l = im_1_fwhm_l/gaussian_sigma_to_fwhm
            
                im_1_fwhm = np.max([im_1_fwhm_l,im_1_fwhm_f])
                im_1_sig = np.max([im_1_sig_l,im_1_sig_f])

            s = find_stars(image=im_data[j], sigma=im_1_sig, ratio=im_1_ratio, theta=im_1_theta, peak=100000)
            stars.append(s)

            fig,ax = plt.subplots(1)

            ax.set_title(imset[j])

            ax.imshow(im_data[j],norm=norm,origin='lower')

            if s is not None:
                for k in range(len(s['xcentroid'])):
                    circle = plt.Circle((s['xcentroid'][k],s['ycentroid'][k]),50,fill=False)#3*im_1_sig_f)
                    ax.add_patch(circle)

            pp.savefig()
            plt.close()

        avg_airmass = np.mean(airmass)

        isot_date_obs = Time(np.mean(jd_middle),format='jd').isot
        
        _,bjd,_ = saph.utils.brvc(isot_date_obs,0.0,tele,ra=tra,dec=tdec)
       

        diffs = []
        for s in range(len(stars)):
            diff = calculate_shift(stars1=stars[0], stars2=stars[s])
            diffs.append(diff)
            #print(diff)

        images = []
        xshift_ints = []
        yshift_ints = []
    
        #this centroid shifting thing is hit or miss, probably more miss. 
        for i in range(len(im_data)):
            image_shift,xshift_int,yshift_int = roll_image(image=im_data[i], diff=diffs[i], threshold=0.5)
            xshift_ints.append(xshift_int)
            yshift_ints.append(yshift_int)
            images.append(image_shift)
        median_image = median_combiner(images=images)

        norm_med = ImageNormalize(median_image, interval=ZScaleInterval(), stretch=SqrtStretch())
        
        print('FWHM fit = ',im_1_fwhm_f)

        print('FWHM LCO median = ',im_1_fwhm_l)


        fig,ax = plt.subplots()

        ax.set_title('FWHM Cutout Image 1: Peak,FWHM,ratio,theta = '+
                     str(np.round(np.max(im_1_star),0))+','+
                     str(np.round(im_1_fwhm_f,0))+','+
                     str(np.round(im_1_ratio,2))+','+
                     str(np.round(im_1_theta*180/np.pi,2)))

        norm_star = ImageNormalize(im_1_star, interval=ZScaleInterval(), stretch=SqrtStretch())
        norm_star = simple_norm(im_1_star, 'linear')

        ax.imshow(im_1_star,origin='lower')

        pp.savefig()
        plt.close()

        size = 30
        im_2_star = im_data[1][r_1-size:r_1+size+1,c_1-size:c_1+size+1]
        im_3_star = im_data[2][r_1-size:r_1+size+1,c_1-size:c_1+size+1]
        im_m_star = median_image[r_1-size:r_1+size+1,c_1-size:c_1+size+1]

        x_cut = np.arange(im_1_star.shape[0])-30
        y_cut = np.arange(im_1_star.shape[1])-30

        X_cut,Y_cut = np.meshgrid(x_cut,y_cut)
        R_cut = np.sqrt(X_cut**2 + Y_cut**2)


        fig,ax = plt.subplots()

        ax.plot(R_cut.ravel(),im_1_star.ravel(),'.')

        ax.plot(R_cut.ravel(),im_m_star.ravel(),'.')

        ax.axvline(im_1_fwhm/2.0)

        pp.savefig()
        plt.close()


        fig,ax = plt.subplots(4,figsize=(6.4,10))

        ax[0].set_title('Cutout 1')

        norm_star = ImageNormalize(im_1_star, interval=ZScaleInterval(), stretch=SqrtStretch())
        norm_star = simple_norm(im_1_star, 'linear')

        ax[0].imshow(im_1_star,origin='lower')
        ax[0].plot(size,size,'o',color='red',fillstyle='none')

        ax[1].set_title('Cutout 2: x='+str(xshift_ints[1])+', y='+str(yshift_ints[1]))
        ax[1].imshow(im_2_star,origin='lower')
        ax[1].plot(size - xshift_ints[1],size - yshift_ints[1],'o',color='red',fillstyle='none')

        ax[2].set_title('Cutout 3: x='+str(xshift_ints[2])+', y='+str(yshift_ints[2]))
        ax[2].imshow(im_3_star,origin='lower')
        ax[2].plot(size - xshift_ints[2],size - yshift_ints[2],'o',color='red',fillstyle='none')

        ax[3].set_title('Shifted and Combined')
        ax[3].imshow(im_m_star,origin='lower')
        ax[3].plot(size,size,'o',color='red',fillstyle='none')


        plt.tight_layout()
        pp.savefig()
        plt.close()

        fig,ax = plt.subplots()
        
        ax.set_title('Combined Image: Pixels')
        
        ax.imshow(median_image,norm=norm_med,origin='lower')
        
        pp.savefig()
        plt.close()


        #This is the aperture photometry verision
        phot_table = aperture_phot(median_image,im_1_fwhm,im_1_ratio,im_1_theta)
              
        cutouts, cut_size = ap_phot_cutouts(median_image, phot_table, im_1_fwhm)



        try:
            sky_web,w_wcsheader,w_web = get_wcs(phot_table)

            sky = w_web.pixel_to_world(phot_table['xcenter'], phot_table['ycenter'])

            min_tdist = np.min(np.sqrt((sky_web.ra.value-tra)**2 + (sky_web.dec.value-tdec)**2))*3600

            if min_tdist > 10:
                print('Target was not found, trying a larger fwhm')

                phot_table = aperture_phot(median_image,im_1_fwhm*2,im_1_ratio,im_1_theta) 
       
                cutouts, cut_size = ap_phot_cutouts(median_image, phot_table, im_1_fwhm)

                sky_web,w_wcsheader,w_web = get_wcs(phot_table)

                sky = w_web.pixel_to_world(phot_table['xcenter'], phot_table['ycenter'])

                min_tdist = np.min(np.sqrt((sky_web.ra.value-tra)**2 + (sky_web.dec.value-tdec)**2))*3600

                if min_tdist > 10:
                    print('Target was still not found...')

                    print('Check the pdf for star locations, '+name_out+'.pdf')
                    
                    nrows = np.min([30,len(cutouts)])
                    ncols = 1
                    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4, 6.4*nrows), squeeze=True)
        
                    cut_size = cutouts[0].shape[0]
        
                    for i in range(0, nrows*ncols):
                        norm = simple_norm(cutouts[i],'linear')
        
                        im = ax[i].imshow(cutouts[i], norm=norm, origin='lower', cmap='viridis')
                        fig.colorbar(im, ax = ax[i])
                        
                        fwhm3_el = patches.Ellipse((cut_size/2,cut_size/2),3*im_1_fwhm,3*im_1_fwhm,angle=0,lw=2,fill=False,color='white')
                        ax[i].add_patch(fwhm3_el)
                        
                        fwhm4_el = patches.Ellipse((cut_size/2,cut_size/2),4*im_1_fwhm,4*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm4_el)
                        
                        fwhm5_el = patches.Ellipse((cut_size/2,cut_size/2),5*im_1_fwhm,5*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm5_el)
        
        
                    plt.tight_layout()
                    pp.savefig()
                    plt.close()
                
        
                    fig,ax = plt.subplots()
        
                    ax.set_title('Combined Image: Pixels')
        
                    ax.imshow(median_image,norm=norm_med,origin='lower')
        
                    ax.plot(phot_table['xcenter'],phot_table['ycenter'],'o',ms=5,fillstyle='none',color='C0')
        
                    pp.savefig()
                    plt.close()

                    f = open(name_out+'_summary.txt', 'w')
                    f.write('Summary: Target was not found in original search or a 2xFWHM search\n')

                    f.write('Output diagnostic pdf: '+name_out+'.pdf\n')
                    f.write('Input Images: '+imset[0]+','+imset[1]+','+imset[2]+'\n')
                    f.write('X Pixel Shift Sizes:'+str(xshift_ints[1])+','+str(xshift_ints[2])+'\n')
                    f.write('Y Pixel Shift Sizes:'+str(yshift_ints[1])+','+str(yshift_ints[2])+'\n')
                    f.write('FWHM fit: '+str(im_1_fwhm_f)+'\n')
                    f.write('FWHM LCO: '+str(im_1_fwhm_l)+'\n')
                    f.write('Stars Found: '+str(len(sky_web.ra.value))+'\n')
                    f.write('Dist From Target: '+str(min_tdist)+'\n')
                    
                    f.close()
                    
                    os.system('cat '+name_out+'_summary.txt')

                else:
                    print('Target found with the 2xFWHM search!')
                    ##################################
                    ###### SUCCESSFUL EXIT CODE ######
                    ##################################
                    nrows = np.min([30,len(cutouts)])
                    ncols = 1
                    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4, 6.4*nrows), squeeze=True)
        
                    cut_size = cutouts[0].shape[0]
        
                    for i in range(0, nrows*ncols):
                        norm = simple_norm(cutouts[i],'linear')
        
                        im = ax[i].imshow(cutouts[i], norm=norm, origin='lower', cmap='viridis')
                        fig.colorbar(im, ax = ax[i])
                        
                        fwhm3_el = patches.Ellipse((cut_size/2,cut_size/2),3*im_1_fwhm,3*im_1_fwhm,angle=0,lw=2,fill=False,color='white')
                        ax[i].add_patch(fwhm3_el)
                        
                        fwhm4_el = patches.Ellipse((cut_size/2,cut_size/2),4*im_1_fwhm,4*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm4_el)
                        
                        fwhm5_el = patches.Ellipse((cut_size/2,cut_size/2),5*im_1_fwhm,5*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm5_el)
        
        
                    plt.tight_layout()
                    pp.savefig()
                    plt.close()
                
        
                    fig,ax = plt.subplots()
        
                    ax.set_title('Combined Image: Pixels')
        
                    ax.imshow(median_image,norm=norm_med,origin='lower')
        
                    ax.plot(phot_table['xcenter'],phot_table['ycenter'],'o',ms=5,fillstyle='none',color='C0')
        
                    pp.savefig()
                    plt.close()
    
    
                    fig = plt.figure() 
                    norm_med = ImageNormalize(median_image, interval=ZScaleInterval(), stretch=SqrtStretch())
                    
                    ax = plt.subplot(projection=w_web)
                    
                    ax.set_title('Combined Image: WCS \n')
                    ax.imshow(median_image,norm=norm_med,origin='lower')
                    
                    ax.plot(sky_web.ra.value,sky_web.dec.value,'o',ms=5,fillstyle='none',transform=ax.get_transform('world'))
                    
                    ax.plot(tra,tdec,'o',ms=10,fillstyle='none',transform=ax.get_transform('world'))
                    
                    ax.coords[0].set_ticklabel_position('bltr')
                    ax.coords[1].set_ticklabel_position('bltr')
                    ax.coords['ra'].set_axislabel('RA')
                    ax.coords['dec'].set_axislabel('Dec')
                    
                    plt.tight_layout()
                    pp.savefig()
        
                    plt.close()
        
    
                    write_csv_apphot(name=name_out+'.csv', im_name=im_1[:22]+'_'+str(nightid_out), bjd=bjd[0], filt=im_1_header['FILTER'], airmass=avg_airmass, results=phot_table, sky=sky_web)
        
                    f = open(name_out+'_summary.txt', 'w')
                    f.write('Summary: Successful run with the 2x FWHM search.\n')
    
                    f.write('Output photometry: '+name_out+'.csv\n')
                    f.write('Output diagnostic pdf: '+name_out+'.pdf\n')
                    f.write('Input Images: '+imset[0]+','+imset[1]+','+imset[2]+'\n')
                    f.write('X Pixel Shift Sizes:'+str(xshift_ints[1])+','+str(xshift_ints[2])+'\n')
                    f.write('Y Pixel Shift Sizes:'+str(yshift_ints[1])+','+str(yshift_ints[2])+'\n')
                    f.write('FWHM fit: '+str(im_1_fwhm_f)+'\n')
                    f.write('FWHM LCO: '+str(im_1_fwhm_l)+'\n')
                    f.write('Stars Found: '+str(len(sky_web.ra.value))+'\n')
                    f.write('Dist From Target: '+str(min_tdist)+'\n')
                    f.close()
                    
                    os.system('cat '+name_out+'_summary.txt')
                    ##################################
                    ##################################
                    ##################################


            else:
                print('Target was found!')

                ##################################
                ###### SUCCESSFUL EXIT CODE ######
                ##################################
                nrows = np.min([30,len(cutouts)])
                ncols = 1
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4, 6.4*nrows), squeeze=True)
        
                cut_size = cutouts[0].shape[0]
    
                for i in range(0, nrows*ncols):
                    norm = simple_norm(cutouts[i],'linear')
        
                    im = ax[i].imshow(cutouts[i], norm=norm, origin='lower', cmap='viridis')
                    fig.colorbar(im, ax = ax[i])
                    
                    fwhm3_el = patches.Ellipse((cut_size/2,cut_size/2),3*im_1_fwhm,3*im_1_fwhm,angle=0,lw=2,fill=False,color='white')
                    ax[i].add_patch(fwhm3_el)
                    
                    fwhm4_el = patches.Ellipse((cut_size/2,cut_size/2),4*im_1_fwhm,4*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                    ax[i].add_patch(fwhm4_el)
                    
                    fwhm5_el = patches.Ellipse((cut_size/2,cut_size/2),5*im_1_fwhm,5*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                    ax[i].add_patch(fwhm5_el)
        
        
                plt.tight_layout()
                pp.savefig()
                plt.close()
        
    
                fig,ax = plt.subplots()
        
                ax.set_title('Combined Image: Pixels')
        
                ax.imshow(median_image,norm=norm_med,origin='lower')
        
                ax.plot(phot_table['xcenter'],phot_table['ycenter'],'o',ms=5,fillstyle='none',color='C0')
        
                pp.savefig()
                plt.close()


                fig = plt.figure() 
                norm_med = ImageNormalize(median_image, interval=ZScaleInterval(), stretch=SqrtStretch())
                
                ax = plt.subplot(projection=w_web)
                
                ax.set_title('Combined Image: WCS \n')
                ax.imshow(median_image,norm=norm_med,origin='lower')
                
                ax.plot(sky_web.ra.value,sky_web.dec.value,'o',ms=5,fillstyle='none',transform=ax.get_transform('world'))
                
                ax.plot(tra,tdec,'o',ms=10,fillstyle='none',transform=ax.get_transform('world'))
                
                ax.coords[0].set_ticklabel_position('bltr')
                ax.coords[1].set_ticklabel_position('bltr')
                ax.coords['ra'].set_axislabel('RA')
                ax.coords['dec'].set_axislabel('Dec')
                
                plt.tight_layout()
                pp.savefig()
    
                plt.close()
    

                write_csv_apphot(name=name_out+'.csv', im_name=im_1[:22]+'_'+str(nightid_out), bjd=bjd[0], filt=im_1_header['FILTER'], airmass=avg_airmass, results=phot_table, sky=sky_web)
        
                f = open(name_out+'_summary.txt', 'w')
                f.write('Summary: Successful run!\n')

                f.write('Output photometry: '+name_out+'.csv\n')
                f.write('Output diagnostic pdf: '+name_out+'.pdf\n')
                f.write('Input Images: '+imset[0]+','+imset[1]+','+imset[2]+'\n')
                f.write('X Pixel Shift Sizes:'+str(xshift_ints[1])+','+str(xshift_ints[2])+'\n')
                f.write('Y Pixel Shift Sizes:'+str(yshift_ints[1])+','+str(yshift_ints[2])+'\n')
                f.write('FWHM fit: '+str(im_1_fwhm_f)+'\n')
                f.write('FWHM LCO: '+str(im_1_fwhm_l)+'\n')
                f.write('Stars Found: '+str(len(sky_web.ra.value))+'\n')
                f.write('Dist From Target: '+str(min_tdist)+'\n')
                f.close()
                
                os.system('cat '+name_out+'_summary.txt')
                ##################################
                ##################################
                ##################################


        except:
            print('Initial WCS solution failed.')
            print('Trying photometry with a larger fwhm')

            phot_table = aperture_phot(median_image,im_1_fwhm*2,im_1_ratio,im_1_theta,threshold=2)
              
            cutouts, cut_size = ap_phot_cutouts(median_image, phot_table, im_1_fwhm)

            try:
                sky_web,w_wcsheader,w_web = get_wcs(phot_table)

                sky = w_web.pixel_to_world(phot_table['xcenter'], phot_table['ycenter'])

                min_tdist = np.min(np.sqrt((sky_web.ra.value-tra)**2 + (sky_web.dec.value-tdec)**2))*3600

                if min_tdist > 10:
                    print('WCS solution worked, but the target was not found.')

                    print('Check the pdf for star locations, '+name_out+'.pdf')
                    
                    nrows = np.min([30,len(cutouts)])
                    ncols = 1
                    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4, 6.4*nrows), squeeze=True)
            
                    cut_size = cutouts[0].shape[0]
        
                    for i in range(0, nrows*ncols):
                        norm = simple_norm(cutouts[i],'linear')
            
                        im = ax[i].imshow(cutouts[i], norm=norm, origin='lower', cmap='viridis')
                        fig.colorbar(im, ax = ax[i])
                        
                        fwhm3_el = patches.Ellipse((cut_size/2,cut_size/2),3*im_1_fwhm,3*im_1_fwhm,angle=0,lw=2,fill=False,color='white')
                        ax[i].add_patch(fwhm3_el)
                        
                        fwhm4_el = patches.Ellipse((cut_size/2,cut_size/2),4*im_1_fwhm,4*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm4_el)
                        
                        fwhm5_el = patches.Ellipse((cut_size/2,cut_size/2),5*im_1_fwhm,5*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm5_el)
            
            
                    plt.tight_layout()
                    pp.savefig()
                    plt.close()
                
        
                    fig,ax = plt.subplots()
            
                    ax.set_title('Combined Image: Pixels')
            
                    ax.imshow(median_image,norm=norm_med,origin='lower')
            
                    ax.plot(phot_table['xcenter'],phot_table['ycenter'],'o',ms=5,fillstyle='none',color='C0')
            
                    pp.savefig()
                    plt.close()


                    f = open(name_out+'_summary.txt', 'w')
                    f.write('Summary: WCS failed in the original search, worked in the 2xFWHM search, but the target was not found.\n')

                    f.write('Output diagnostic pdf: '+name_out+'.pdf\n')
                    f.write('Input Images: '+imset[0]+','+imset[1]+','+imset[2]+'\n')
                    f.write('X Pixel Shift Sizes:'+str(xshift_ints[1])+','+str(xshift_ints[2])+'\n')
                    f.write('Y Pixel Shift Sizes:'+str(yshift_ints[1])+','+str(yshift_ints[2])+'\n')
                    f.write('FWHM fit: '+str(im_1_fwhm_f)+'\n')
                    f.write('FWHM LCO: '+str(im_1_fwhm_l)+'\n')
                    f.write('Stars Found: '+str(len(sky_web.ra.value))+'\n')
                    f.write('Dist From Target: '+str(min_tdist)+'\n')
                    
                    f.close()
                    
                    os.system('cat '+name_out+'_summary.txt')


                else:
                    print('WCS solution worked, and the target was found!')
                    ##################################
                    ###### SUCCESSFUL EXIT CODE ######
                    ##################################
                    nrows = np.min([30,len(cutouts)])
                    ncols = 1
                    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4, 6.4*nrows), squeeze=True)
            
                    cut_size = cutouts[0].shape[0]
        
                    for i in range(0, nrows*ncols):
                        norm = simple_norm(cutouts[i],'linear')
            
                        im = ax[i].imshow(cutouts[i], norm=norm, origin='lower', cmap='viridis')
                        fig.colorbar(im, ax = ax[i])
                        
                        fwhm3_el = patches.Ellipse((cut_size/2,cut_size/2),3*im_1_fwhm,3*im_1_fwhm,angle=0,lw=2,fill=False,color='white')
                        ax[i].add_patch(fwhm3_el)
                        
                        fwhm4_el = patches.Ellipse((cut_size/2,cut_size/2),4*im_1_fwhm,4*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm4_el)
                        
                        fwhm5_el = patches.Ellipse((cut_size/2,cut_size/2),5*im_1_fwhm,5*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                        ax[i].add_patch(fwhm5_el)
            
            
                    plt.tight_layout()
                    pp.savefig()
                    plt.close()
                
        
                    fig,ax = plt.subplots()
            
                    ax.set_title('Combined Image: Pixels')
            
                    ax.imshow(median_image,norm=norm_med,origin='lower')
            
                    ax.plot(phot_table['xcenter'],phot_table['ycenter'],'o',ms=5,fillstyle='none',color='C0')
            
                    pp.savefig()
                    plt.close()

                    fig = plt.figure() 
                    norm_med = ImageNormalize(median_image, interval=ZScaleInterval(), stretch=SqrtStretch())
                    
                    ax = plt.subplot(projection=w_web)
                    
                    ax.set_title('Combined Image: WCS \n')
                    ax.imshow(median_image,norm=norm_med,origin='lower')
                    
                    ax.plot(sky_web.ra.value,sky_web.dec.value,'o',ms=5,fillstyle='none',transform=ax.get_transform('world'))
                    
                    ax.plot(tra,tdec,'o',ms=10,fillstyle='none',transform=ax.get_transform('world'))
                    
                    ax.coords[0].set_ticklabel_position('bltr')
                    ax.coords[1].set_ticklabel_position('bltr')
                    ax.coords['ra'].set_axislabel('RA')
                    ax.coords['dec'].set_axislabel('Dec')
                    
                    plt.tight_layout()
                    pp.savefig()
        
                    plt.close()        
    #
                    write_csv_apphot(name=name_out+'.csv', im_name=im_1[:22]+'_'+str(nightid_out), bjd=bjd[0], filt=im_1_header['FILTER'], airmass=avg_airmass, results=phot_table, sky=sky_web)
            
                    f = open(name_out+'_summary.txt', 'w')
                    f.write('Summary: Successful run with a 2xFWHM search.\n')
    #
                    f.write('Output photometry: '+name_out+'.csv\n')
                    f.write('Output diagnostic pdf: '+name_out+'.pdf\n')
                    f.write('Input Images: '+imset[0]+','+imset[1]+','+imset[2]+'\n')
                    f.write('X Pixel Shift Sizes:'+str(xshift_ints[1])+','+str(xshift_ints[2])+'\n')
                    f.write('Y Pixel Shift Sizes:'+str(yshift_ints[1])+','+str(yshift_ints[2])+'\n')
                    f.write('FWHM fit: '+str(im_1_fwhm_f)+'\n')
                    f.write('FWHM LCO: '+str(im_1_fwhm_l)+'\n')
                    f.write('Stars Found: '+str(len(sky_web.ra.value))+'\n')
                    f.write('Dist From Target: '+str(min_tdist)+'\n')
                    f.close()
                    
                    os.system('cat '+name_out+'_summary.txt')
                    ##################################
                    ##################################
                    ##################################



            except:
                print('Second WCS solution failed.')

                print('Check the pdf for star locations, '+name_out+'.pdf')
                
                nrows = np.min([30,len(cutouts)])
                ncols = 1
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4, 6.4*nrows), squeeze=True)
            
                cut_size = cutouts[0].shape[0]
        
                for i in range(0, nrows*ncols):
                    norm = simple_norm(cutouts[i],'linear')
            
                    im = ax[i].imshow(cutouts[i], norm=norm, origin='lower', cmap='viridis')
                    fig.colorbar(im, ax = ax[i])
                    
                    fwhm3_el = patches.Ellipse((cut_size/2,cut_size/2),3*im_1_fwhm,3*im_1_fwhm,angle=0,lw=2,fill=False,color='white')
                    ax[i].add_patch(fwhm3_el)
                    
                    fwhm4_el = patches.Ellipse((cut_size/2,cut_size/2),4*im_1_fwhm,4*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                    ax[i].add_patch(fwhm4_el)
                    
                    fwhm5_el = patches.Ellipse((cut_size/2,cut_size/2),5*im_1_fwhm,5*im_1_fwhm,angle=0,lw=2,fill=False,color='red')
                    ax[i].add_patch(fwhm5_el)
            
            
                plt.tight_layout()
                pp.savefig()
                plt.close()
                
        
                fig,ax = plt.subplots()
            
                ax.set_title('Combined Image: Pixels')
            
                ax.imshow(median_image,norm=norm_med,origin='lower')
            
                ax.plot(phot_table['xcenter'],phot_table['ycenter'],'o',ms=5,fillstyle='none',color='C0')
            
                pp.savefig()
                plt.close()

                f = open(name_out+'_summary.txt', 'w')
                f.write('Summary: WCS failed in original search and the 2xFWHM search\n')

                f.write('Output diagnostic pdf: '+name_out+'.pdf\n')
                f.write('Input Images: '+imset[0]+','+imset[1]+','+imset[2]+'\n')
                f.write('X Pixel Shift Sizes:'+str(xshift_ints[1])+','+str(xshift_ints[2])+'\n')
                f.write('Y Pixel Shift Sizes:'+str(yshift_ints[1])+','+str(yshift_ints[2])+'\n')
                f.write('FWHM fit: '+str(im_1_fwhm_f)+'\n')
                f.write('FWHM LCO: '+str(im_1_fwhm_l)+'\n')
                f.write('Stars Found: '+str(len(phot_table['xcenter']))+'\n')
                
                f.close()
                
                os.system('cat '+name_out+'_summary.txt')


        pp.close()

    print('')


