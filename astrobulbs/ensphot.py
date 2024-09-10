import numpy as np
import os
import time
from astropy.time import Time
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import curve_fit
from scipy.signal import spectral
from scipy import interpolate
import astropy.io.fits as fits
import pickle
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy.ndimage import gaussian_filter

#version 0.1.0 - 2021/07/22

def ens_match(in_file,tra,tdec,scope='lcogt',out_file='default',d_max=1.0,nn_dist=20,edge_buffer=50, min_apps=30,mag_ulim=50.0,mag_llim=-50.0,magerr_ulim=0.5):

    '''
    A program to find matching stars in each image, from a photometry ouput, to 
    then be used for ensemble differential photometry. 

    This program is designed to be run after data_shred or pyphot. The output
    for either program is a '.dat' file that contains relavent photometry
    fields and the RA and Dec locations. The ideal input for this program is 
    all of the .dat files (one for each image) put into one file. You can do 
    this with a simple:
    'cat *.dat > all.dat' (or some such)

    If you are running this program without running data_shred or pyphot you 
    can create the necessary file in the following way. Note: Images must have
    a WCS solution file. 
    --------------------------------------------------------------------------
    txdump f_name+'.mag.1' fields='xc,yc' expr='yes' Stdout='in.radec.coo'
    -> This pulls out the x, y pixel values for a given file

    cctran 'in.radec.coo' output='out.radec.coo' database=f_name+'coords.db' 
           solution=f_name forward='yes' lngformat='%13.7g' latformat='%13.7g'
           lngunits = "degrees" latunits = "degrees"
    -> This makes a list of the RA and Dec for the x, y pixel values for a 
       given file.

    txdump textfiles=f_name+'.mag.1' fields='image,id,otime,flux,mag,merr,
           ifilt,perror,xc,yc,xair' expr='yes' Stdout='pre.dat'
    -> This makes the .dat file with all but the RA, Dec coordinates.

    cat pre.dat | sed s/INDEF/9999.9/g > pre.dat
    -> Get rid of bad data
    
    paste pre.dat out.radec.coo > f_name'.dat'
    -> "paste" on two extra columns that are the RA and Dec.

    If you have used py.sexy_phot all you have to do it compile all of the 
    ouput photometry into one file. 
    -> cat 2*.dat.sex > raw_phot.dat.sex #for instance
    --------------------------------------------------------------------------
    
    The program first finds the target in each image within some user 
    specified tolerance. The program reports how how far it had to search for
    the target and how many frames it found the target in. 

    At this point you have the option to quit and change the tolerances if
    results are poor or accept them and match comparison stars from each 
    image. 

    The result is saved as a dictionary in pickle format.

    Program want there to be a t_redec.coo file containing the RA and Dec of 
    target in decimal degrees. Same a py.data_shred or py.data_shred_hc.

    The output matched data dictionary is structured as follows:
    -> Star ID (is either a number padded with four zeros of "Target")
       -> Array containing information for matched stars in the following order:
          -> [0] image name
          -> [1] id in image
          -> [2] hjd
          -> [3] flux
          -> [4] fluxerr 		#previously not included
          -> [5] mag 		   	#previously [4]
          -> [6] magerr 		#previously [5]
          -> [7] filter			#previously [6]
          -> [8] image x pixel location
          -> [9] image y pixel location
          -> [10] ra
          -> [11] dec
          -> [12] weight 1: Global weight for this image (1.0 or 0.0)
          -> [13] weight 2: Global weight for this star (1.0 or 0.0)
          -> [14] weight 3: Star's weight in this image (1.0 or 0.0)
          -> [15] weight 4: Star's weight (1/sigma^2) in this image 
          -> [16] airmass
   
    Parameters:
    -----------
    in_file: str
        File containing all of the photometry information from each image. 
        See description above. 

    scope : str
        Telescope the data are coming from. The default is 'lcogt'. 
        Acceptable values are 'lcogt', 'lcogtnb', 'smarts', 'smartsir', 'hdi', 
        and 'arc'. 

    out_file: str
        Output file name for the matched dictionary pickle.

    d_max: value 
        Maximum allowed distance between first instance of a star and its 
        match. Default=1"
        ARCSECONDS

    nn_dist: value
        Minimum distance required between nearest neighbor for a match 
        to be made. In place to insure comparisons stars are not blended. 
        Default=20 pixels
        PIXELS 

    min_apps: value
        Minimum number of frames matched for a star to be used for 
        ensemble photometry. 
        Default=30
        NUMBER

    mag_ulim: value
        Upper limit for acceptable instrumental magnitudes. Default=25.0 mags.
        MAGNITUDES

    mag_llim: value
        Lower limit for acceptable instrumental magnitudes. Default=5.0 mags.
        MAGNITUDES

    magerr_ulim: value
        Upper limit on the acceptable error in an instrumental magnitude.
        Default=0.5
        MAGNITUDES

    Returns:
    --------
    N/A

    Output:
    -------
    - Dictionary of matched stars in pickle format. 

    Version History:
    ----------------
    2015-06-25 - Bug Fix: Was registering the same star mutiple times. Don't 
                 trust photometry before this date.
    2015-07-06 - Added in S2KB compatability, consolidated some of the
                 conditional statements.
    2015-07-20 - Added in airmass to dictionary structure. 
                 Added HDI compatibility.
    2015-07-30 - Sped up the matching process by:
                 1) only searching forward
                 2) not searching over previously matched stars
                 3) not starting new searches on images less than min_apps from 
                    the end of the image list
                 Set the throw away weight from 10**-8 to 10**-20 seems to make
                 no difference.
    2015-08-10 - Bug Fix: Needed conditional statements to handle the case when
                 all stars were found in an image. Idecies were messing up. 
    '''

    while scope not in ['arc','lcogt','lcogtnb','smarts','smartsir','s2kb',
                        'hdi','hdifov']:
        print('Incorrect telescope specified. Try again')
        print('Acceptable values are: arc, lcogt, smarts, smartsir, s2kb, hdi,')
        print('and hdifov')
        scope = input('-->')

    if scope == 'lcogt':
        pixs=0.389
        px_max = 4096
        py_max = 4096

    if scope == 'lcogtnb':
        pixs=0.232

    if scope == 'smarts':
        pixs=0.371

    if scope == 'arc':
        pixs=1.312
        px_max=512.0
        py_max=512.0

    if scope == 's2kb':
        pixs=1.19
        px_max=544.0
        py_max=512.0 

    if scope == 'hdi':
        pixs=0.43
        px_max=2048.0
        py_max=2056.0 

    if scope == 'hdifov':
        pixs=0.43
        px_max=4500.0
        py_max=4500.0 

    if scope == 'smartsir':
        pixs=0.276
        px_max=712.0
        py_max=712.0

    fname,iid,hjd,flux,fluxerr,mag,magerr,\
        filt,px,py,air,ra,dec=np.loadtxt(in_file,unpack=True,delimiter=',',
                                         dtype='U100,f,float64,f,f,f,f,U10,f,\
                                         f,f,float64,float64')
    
    print('Number of frames:',np.unique(fname).size)

    g=((py > edge_buffer) & 
       (px > edge_buffer) & 
       (mag < mag_ulim) & 
       (magerr < magerr_ulim) & 
       (magerr > 0) & 
       (mag > mag_llim) &
       (py < py_max-edge_buffer) &
       (px < px_max-edge_buffer))

    print('Number of frames that pass the cuts:',np.unique(fname).size)

    print('Numebr of stars passing the cut across the whole data set:',np.sum(g))

    fname=fname[g]
    iid=iid[g]
    hjd=hjd[g]
    flux=flux[g]
    fluxerr=fluxerr[g]
    mag=mag[g]
    magerr=magerr[g]
    filt=filt[g]
    px=px[g]
    py=py[g]
    ra=ra[g]
    dec=dec[g]
    air=air[g]

    ee=np.unique(fname)

    print(ee.size)
        
    matches={'Target':[]}
    bad_frames=np.empty(0)

    print('')
    print('Target Distace Match ('+str(d_max)+' arcsec = '+str(d_max/3600)+')')

    #A mask to exclude stars that have already been found.
    found=np.zeros(fname.size)

    for i in range(ee.size):
        m_list = (fname == ee[i])

        d=np.sqrt((tra-ra[m_list])**2+(tdec-(dec[m_list]))**2)
        
        print(np.min(d)*3600, ee[i])
        
        tar_match=(d == np.min(d))
        if np.min(d) <= d_max*0.00027:
            w = [1.0, 1.0, 1.0, 1.0/(((magerr[m_list])[tar_match])[0])**2]
            matches['Target'].append([((fname[m_list])[tar_match])[0],
                                      ((iid[m_list])[tar_match])[0],
                                      ((hjd[m_list])[tar_match])[0],
                                      ((flux[m_list])[tar_match])[0],
                                      ((fluxerr[m_list])[tar_match])[0],
                                      ((mag[m_list])[tar_match])[0],
                                      ((magerr[m_list])[tar_match])[0],
                                      ((filt[m_list])[tar_match])[0],
                                      ((px[m_list])[tar_match])[0],
                                      ((py[m_list])[tar_match])[0],
                                      ((ra[m_list])[tar_match])[0],
                                      ((dec[m_list])[tar_match])[0]]+w+
                                      [((air[m_list])[tar_match])[0]])
            
            found[((fname == ee[i]) & (iid == ((iid[m_list])[tar_match])[0]))] = 1
        
        else:
            bad_frames=np.append(bad_frames,ee[i])

    print('Matched ',ee.size-bad_frames.size,'out of',ee.size,' frames')
    if bad_frames.size >= 1:
        print('Frames with no match: ',bad_frames)
    #else:
        #bad_frames = ['none']
    print('Continue? Yes (y), No (n)?')
    cont=input('-->')
    while cont not in ['y','n']:
        print('Try Again')
        cont=input('-->')
    
    #only continue with frames where the target was found.
    bad_frame_mask = np.ones(ee.size,dtype='bool')
    for i in range(bad_frames.size):
        bad_frame_mask = bad_frame_mask*~(ee == bad_frames[i])

    ee = ee[bad_frame_mask]

    if cont == 'y':
        print('Continuing with '+str(ee.size)+' frames')
        
        for i in range(ee.size-min_apps):
            #I subtract min_apps because if you find a new star that hasn't 
            #been matched in the previous images, even if you match it in each
            #sequential image, it won't be enough to meet the min_apps 
            #criterion.                   
            #i is for each unique image

            to_m = ((fname == ee[i]) & (found == 0.0))
            #to_m, to match, is a mask that selects unfound stars in the current image
            #updated for every image
            
            if np.sum(to_m) > 0:
                #If there are things to match, continue

                for j in range((iid[to_m]).size):
                    #for each star in exposure ee
                    keys=matches.keys()

                    #if ((iid[to_m])[j]) not in [((matches[d])[i])[1] for d in keys]:
                    #   #This conditional doesn't allow the Target star to 
                    #   #be used as a comparison star
                    #   #I think this is redundant, but I'm leaving it in for now

                    if len(matches.keys()) == 1:
                        m_name='0001'
                    else:
                        keys=np.array(list(matches.keys()))
                        run_ind=np.max(np.array(keys[keys != 'Target'],
                                                dtype=float))
                        m_name=str(int(run_ind+1)).zfill(4)

                    t_m={m_name:[]}
                    #Create a dictionary for this star

                    found_id=np.empty(0,dtype=int)
                        #Stores the indecies of matched stars that no 
                        # longer need to be searched over. Updates the found mask below.

                    for l in range(i):
                    	#this for loop adds dictionary rows for previous images
                    	#where this star was not found. If this is the first image
                    	#it does nothing.
                        w = [1.0, 1.0, 1.0*10**-20, 1.0*10**-20]
                    
                        t_m[m_name].append([\
                                'NotFound',
                                0,
                                hjd[fname == ee[l]][0],
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                filt[fname == ee[l]][0],
                                px[fname == ee[l]][0],
                                py[fname == ee[l]][0],
                                ra[fname == ee[l]][0],
                                dec[fname == ee[l]][0]]+w+
                                [air[fname == ee[l]][0]])
                    
                    for k in range(ee.size-i):
                    	#this for loop places data rows for the current image
                    	# and searches subsequent images. 
                       	#Subtract 'i' because you don't need to
                       	#match backwards. 
                        k=k+i

                        if k == i:
                        	#i.e. this image, append data for this image
                            w = [1.0, 1.0, 1.0, 1.0/((magerr[to_m])[j])**2]

                            t_m[m_name].append([(fname[to_m])[j],
                                                (iid[to_m])[j],
                                                (hjd[to_m])[j],
                                                (flux[to_m])[j],
                                                (fluxerr[to_m])[j],
                                                (mag[to_m])[j],
                                                (magerr[to_m])[j],
                                                (filt[to_m])[j],
                                                (px[to_m])[j],
                                                (py[to_m])[j],
                                                (ra[to_m])[j],
                                                (dec[to_m])[j]]+w+
                                               [(air[to_m])[j]])

                            found_id=np.append(found_id,np.where([(fname == (fname[to_m])[j]) & (iid == (iid[to_m])[j])])[1][0])
                        
                        if k > i:
                        	#i.e. subsequent images

                            m_list=((fname == ee[k]) & (found == 0.0))
                            #list of stars that have not yet been matched in the subsequent image
                            #local mask for the next image
                            
                            if np.sum(m_list) > 0: #i.e. if there are stars in this image that have not been matched.

                                d=np.sqrt(((ra[to_m])[j] - ra[m_list])**2 + ((dec[to_m])[j] - dec[m_list])**2)

                                d_add=np.append(d,100.0)

                                c_match=(d == np.min(d))

                                if ((np.min(d) < d_max/3600.0) &						#distance is less than the threshold
                                    ((np.sort(d_add))[1] > nn_dist*(pixs/3600.0))		#nearest neightbor is not too close
                                    )==True:

                                	#if conditions are met for this row, add its photometry data to the dictionary

                                    w = [1.0, 1.0, 1.0, 1.0/(((magerr[m_list])[c_match])[0])**2]

                                    found_id=np.append(found_id,np.where([(fname == ((fname[m_list])[c_match])[0]) & (iid == ((iid[m_list])[c_match])[0])])[1][0])
                                    
                                    t_m[m_name].append( \
                                        [((fname[m_list])[c_match])[0],
                                         ((iid[m_list])[c_match])[0],
                                         ((hjd[m_list])[c_match])[0],
                                         ((flux[m_list])[c_match])[0],
                                         ((fluxerr[m_list])[c_match])[0],
                                         ((mag[m_list])[c_match])[0],
                                         ((magerr[m_list])[c_match])[0],
                                         ((filt[m_list])[c_match])[0],
                                         ((px[m_list])[c_match])[0],
                                         ((py[m_list])[c_match])[0],
                                         ((ra[m_list])[c_match])[0],
                                         ((dec[m_list])[c_match])[0]]+w+
                                        [((air[m_list])[c_match])[0]])
                                else:
                                	#if not, add not found values
                                    w = [1.0, 1.0 ,1.0*10**-20, 1.0*10**-20]

                                    t_m[m_name].append([\
                                            'NotFound',
                                            0,
                                            hjd[fname == ee[i]][0],
                                            0.1,
                                            0.1,
                                            0.1,
                                            0.1,
                                            filt[fname == ee[i]][0],
                                            px[fname == ee[i]][0],
                                            py[fname == ee[i]][0],
                                            ra[fname == ee[i]][0],
                                            dec[fname == ee[i]][0]] +w+ \
                                            [air[fname == ee[i]][0]])

                            else:	#i.e. there are no stars left in this image that have not already been matched, i.e., the star is not found.

                                w = [1.0, 1.0, 1.0*10**-20, 1.0*10**-20]

                                t_m[m_name].append([\
                                        'NotFound',
                                        0,
                                        hjd[fname == ee[i]][0],
                                        0.1,
                                        0.1,
                                        0.1,
                                        0.1,
                                        filt,
                                        px[fname == ee[i]][0],
                                        py[fname == ee[i]][0],
                                        ra[fname == ee[i]][0],
                                        dec[fname == ee[i]][0]]+w+
                                        [air[fname == ee[i]][0]])

                        if ((i == 0) & (j == 1)):
                            print(k,fname[(fname == ee[k])][0])

                    #print(t_m[m_name][0])
                    if np.sum([row[14] for row in t_m[m_name]]) >= min_apps:
                        matches[m_name]=t_m[m_name]
                        found[found_id] = 1                            
                                                                           
            print(float(i)/(ee.size-min_apps))

        keys=np.array(list(matches.keys()))
        
        print('')
        print('Total comp stars matched: ',np.array(keys[keys != 'Target'],
                                              dtype=str).size)
        print('')

        pickle.dump(matches,open(out_file+'.p','wb'))

        name=np.array(['in_file ','scope   ','out_file','d_max   ','nn_dist ',
                       'min_apps','mag_ulim','mag_llim','magerr_ulim'])
        used=np.array([in_file,scope,out_file,d_max,nn_dist,min_apps,mag_ulim,
                       mag_llim,magerr_ulim],dtype='str')
        np.savetxt(out_file+'.mpar',np.c_[name,used],fmt='%s')
    
        keys=np.array(list(matches.keys()))
    
        ss=np.array(keys,dtype=str)
        ss_eff=np.zeros(ee.size) #num of non-zero (10^-8) weight stars. 
        for i in range(ee.size):
            k=0
            for j in range(ss.size):
                if (matches[ss[j]][i][12] == 1.0) &\
                   (matches[ss[j]][i][13] == 1.0) &\
                   (matches[ss[j]][i][14] == 1.0) == True:
                    k=k+1
            ss_eff[i]=k
            print(ee[i],k)


    return


def ens_match_map(infile,tra,tdec):

    '''
    A program to graphically show the RA and Dec location of your target and
    comparison stars as found in the program pyred.ens_match. 

    Upon running this program a graphics window will pop up plotting the RA and
    Dec of your target and comparison stars. This is useful because when 
    running pyred.ens_lc, it is often useful to know which starts you are 
    deleting from the ensemble lightcurve solution. 

    Currently this program does not take into account the curvature of the sky
    but my FOVs are pretty small (~15'). 

    Each registered star, comparison or target, is surrounded by a circle that
    represents the standard deviation in the RA and Dec when matching that star.
    You may have to zoom way in to see it. 

    The user supplied location of the target star is shown as a green 'x'.

    Parameters:
    -----------
    infile: str
        The name of the pickeled star-match dictionary output from ens_matches.

    tfile: str
        Name of a text file containing the RA and Dec of the target star in 
        decimal degrees. 

    Returns:
    --------
    N/A

    Output:
    -------
    N/A
    
    Version History:
    ----------------
    2015-07-14 - Start
    2016-12-28 - Added capability to highlight all of the locations in the 
                 input target list, rather than just one. 
    '''

    matches=pickle.load(open(infile,'rb'))
    keys=list(matches.keys())

    ra=np.zeros(len(keys))
    ra_std=np.zeros(len(keys))
    comp=np.zeros(len(keys))
    dec=np.zeros(len(keys))
    dec_std=np.zeros(len(keys))

    for i in range(len(keys)):
        g=np.array([row[14] for row in matches[keys[i]]]) > 0.1

        ra_i=np.array([row[10] for row in matches[keys[i]]])
        ra[i]=np.mean(ra_i[g])
        ra_std[i]=np.std(ra_i[g])

        dec_i=np.array([row[11] for row in matches[keys[i]]])
        dec[i]=np.mean(dec_i[g])
        dec_std[i]=np.std(dec_i[g])

        if matches[keys[i]][0][13] == 1:
            comp[i]=1

    #print ra,ra_std,dec,dec_std
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(tra,tdec,'x',ms=8,mew=5,color='green')
    ax.plot(ra[comp==1],dec[comp==1],'o',ms=5,color='blue')
    ax.plot(ra[comp==0],dec[comp==0],'o',ms=5,color='red')
    for i in range(ra.size):
        ell = patches.Ellipse((ra[i],dec[i]),width=ra_std[i], height=dec_std[i],
                      		  angle=0.0,fill=False)
        ax.add_patch(ell)
        ax.text(ra[i],dec[i],keys[i])
    ax.axis([np.min(ra)-1.0/60.0,np.max(ra)+1.0/60.0,
             np.min(dec)-1.0/60.0,np.max(dec)+1.0/60.0])
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('Match Results')
    plt.show()
    
    wait=input('-->')
    plt.close(fig)

    return


def ens_lc(in_file,target,date,scope,all_lc='yes',n_min=20,s_width=0.25):

    '''
    An interactive program to compute the ensemble lightcutve of a target star 
    following methodology described in Honeycutt 1992. 

    This program allows you to find the best ensemble of stars and images for 
    differential photometry through three plot interfaces. 

    Plot I: Mean Magnitude Plot
    A plot of the mean magnitude for each star vs its standard deviation. 
    Outliers in this plot will stand out as having large standard devaitions 
    given their magnitude. The "target" stars is included in this plot as the
    red point. If it truely is a variable star, you will want to delete it. Its
    lightcurve will still be output at the end.
    If a star has a circle around it is appears in less frames than you are asking.
    Commands: d - Delete star from ensemble.
              u - Undelete star from ensemble.
              e - Enter Plot II: Lightcurve Plot
                  You may only need to delete one data point instead of the 
                  whole star. This can be done in the Lightcurve Plot
              w - Enter Plot III: Exposure Magnitude Plot
                  A bad exposure may be messing everything up. You can delete 
                  bad exposures in the Exposure Magitude Plot.

    Plot II: Lightcurve Plot
    A plot of a star's observed lightcurve. Bad matches will appear as points 
    way out of line with the rest. 
    Commands: d - Delete nearest data point to cursor from star's lightcurve.
              o - Delete data point furthest from the data mean. 
              u - Undelete data point from star's lightcurve.
              b - Return to Plot I: Mean Magnitude Plot

    Plot III: Exposure Magnitude Plot
    Exposures of low signal-to-noise will appear way off with large error bars. 
    Commands: d - Delete exposure from ensemble.
              u - Undelete exposure from ensemble.
              b - Return to Plot I: Mean Magnitude Plot

    Deleting stars/data points/exposures does not remove the information, it 
    just sets it's weight to 10^(-20).

    Hitting return in the python terminal will exit the plot mode and ask if 
    you would like to recompute the magnitudes. 

    Once satisfied, the target lightcurve is displayed and you are asked if 
    you would like to output the results. 

    Best Practices
    --------------
    Getting the best set of lightcurves from ensemble photometry is a bit of
    an art. Most of the work will be done in Plot I and Plot II where you have
    to make the choice of weather to delete a star from the ensemble (Plot I)
    or just delete some points from within that star's lightcurve (Plot II).
    Because a star's weight in the ensemble depends on its brightness, a lot
    of care needs to go into making sure the brightest stars are first, not
    varying and second, don't contain bad observations. Below I break down my
    best practices into three ordered steps:

    1) Delete The Target Star:
       If you ar going to make some statement about how your target star is
       varying you need to delete it from the ensemble. If it falls right in
       the mix with a bunch of non-varying stars of the same magnitude then,
       to the precision of this method, your star is not varying...
    2) Check Bright Stars:
       If the brightest star in your field of view is variable (and it is much
       brighter than the next star) the program will try to make this star's
       brightness constant, messing up the lightcurves for all the other
       stars.  Start by deleting the brightest stars and hitting return in the
       terminal to see how it effects the rest of the stars. The standard
       deviation of the bright stars will increase when you delete them but as
       long as they don't go up too much, they should be good to keep. It can 
       be good to delete one star at a time and see how things change, big
       changes should raise red flags. We will revisit these stars in step 4.
    3) General Clean Up:
       If things are behaving well, there should be a trend of increasing
       devation with dimmer stars (larger magnitude). There will also likely
       be a few stars off in no-man's land. These could be variable stars
       (your target stars is likely one of these, marked as a red circle) or
       they could be non-varying stars with one bad observations (bad-pixel,
       CR, etc.). If the latter is the case, it would be good to retain this
       star if possible.  Enter the lightcurve plot (Plot II) for each of
       these stars. If they are variable, delete the star in Plot I. If they
       just have a bad observation, delete that individual point in Plot II
       and re-compute the values. If that star now falls in line, keep it, if
       not, delete it.  
    4) Detailed Lightcurve Checks:
       Starting with the brightest star, look through each star's light curve
       to delete observations that look like outliers. You want to be careful
       here. The stars variations will be a combination of the measurement
       precision and the star's underlying variability. If all the points fall
       within an envelope then it's probably fine assuming it's y-location in
       Plot I is agreeable. It can be tempting to delete anything outside the
       two-sigma deviation line but in the end it won't make a ton of
       difference in the final lightcurve. Of course, with a really small
       number of stars these choices become more important as they will have a
       larger effect on the final outcome. With a lot of stars, the ensemble
       will provide you with some cushion.
    
    Parameters 
    ---------- 
    in_file: str 
        The name of the pickeled star-match dictionary output from ens_matches.

    target: str
        Short name for star to be printed in name of output files. 

    date: str
        Date in yyyymmdd format to be printed in output file names. 

    scope: str
        Name of telescope or instrument to be printed in out file names.

    all_lc: str; optional
        If 'yes', creates lightcurve file for each matched star, otherwise it 
        does not. Default is 'no'

    n_min : float, optional
        Sets the nominal number of times a star has to appear within a data
        set. The default values is 20.

    bx_width : float, optional
        Sets the width of the boxcar smoothing kernel for the Mean Magnitude 
        Plot, Plot I. The default value is 0.5 mags.
        
    Returns
    -------
    N/A

    Outputs
    -------
    - Text file with Target lightcurve data. 
    - Directory named 'ens_data' containing the following files:
      - Dictionary with match/weight information used in final lightcurve 
        (pickle).
      - Text file with position and magnitude information for Target and 
        comparison stars. Target is listed as '0'.
      - Histogram of the number of comparison stars used per image.
      - Text file with comparison star lightcurve data. 
    
    Version History
    ---------------
    2015-05-03 - Fixed error estimate. 
    2015-06-24 - Now plots the lightcurve of the ensemble magnitude rather than
                 the image magnitude for comparison stars in PLOT II. Should
                 make it easier to tell when a comparison star is bad.
    2015-06-26 - Lots of changes, nothing is good before this date:
                 - Outputs target LC in valuable units for "absolute" scaling.
                 - Creates an "ens_data" directory with subdirectories for each
                   date the program is run (you set the date). It contains:
                   - Histogram of the number of comp stars used per image.
                   - Lightcurves for all stars.
                   - Mean magnitudes and errors for all comp stars used. 
                   - The final dictionary used to do ens. phot. in a pickle.
    2015-07-20 - Now outputs airmass in the lc files. 
               - Imporvement to histogram. 
    2015-08-19 - Added Target's information to the comp output. 
               - Updates to documentation. 
    2015-11-04 - Changes to ens_fill now include the target star in PLOT I. It
                 will be the red point. 
    2015-11-19 - Changes a write statement to print the standard devation of the
                 RA and Dec in Sci. Notation. Files made before this date list
                 only the sig-figs, they are actually that time 1e-5. Not a 
                 big deal beause I never really use that number. 
    2016-12-07 - Changed Plot II to show the acutal magnitude and magnitude 
                 error that would be output, along with a line for the data 
                 mean, the 2-sigma varience based on the average error, and the
                 2-sigma mean based on the data's standard deviation. 
                 The plot now also naturally zooms to regions of user accepted 
                 data. 
               - A new "best practices" section was added to the documentation. 
               - Updated the average error output to be the average error for 
                 the target star rather than the average error on the exposure.
               - Added 'o' button to Plot II.
               - Updated the output files to include more information about the 
                 photometric precision.
    2016-12-14 - Plot II now removes point if the entire exposure is thrown out.
               - Gives the number of valid points in the top of Plot II
               - Added the n_min parameter.
               - Circles points in Plot I that have less than the nominal 
                 number set with the n_min parameter.
               - Lines on Plot I that show the location of the Target's mean 
                 error.
    2016-12-15 - Added a boxcar-smoothed line of the empirical error curve.
    2016-12-28 - Changed target internal error lines to a green triangle.
               - Added the boxcar smoothing width, bx_width, to the list of 
                 input parameters. 
    2016-12-29 - The extra photometry information output in the enscomps file
                 from 2016-12-07 messed up the file format needed for the
                 stand_johnson program. To ensure that old files, as well as 
                 more recent ones, work, I have moved this information to the 
                 end of each column. 
    '''
    plt.ion()

    matches=pickle.load(open(in_file,'rb'))

    keys,ee,ss,a,b,mag_es,w_es,x,em,emerr,m0,m0err=ens_fill(matches)
    # pyred.ens_fill() can be found below. It is a program to create the 
    # system of linear equations (matrix) from the pyred.ens_match() 
    # dictionary and solve them in the Honeycutt 1992 formalism.

    plotted=['mags','lc','em']
    # Tell press key function what it's doing

    p=np.array([((matches[ss[i]])[0])[13] for i in range(ss.size)],dtype=int)

    mag_shift = np.min(m0[p==1])-1

    m0 = m0 - mag_shift

    # Place holder index for PLOT I
    lc_p=np.ones(ee.size,dtype=int)
    # Place holder index for PLOT II
    em_p=np.array([((matches[ss[0]])[i])[12] for i in range(ee.size)],dtype=int)
    # Place holder index for PLOT III

    lc_mag=np.ones(ee.size,dtype='float64')
    lc_magerr=np.ones(ee.size,dtype='float64')
    lc_hjd=np.ones(ee.size,dtype='float64')
    lc_id=['0000']
    # Initial place holders from PLOT II

    def plot_mags(m0,m0err,p,n_valid,target_err):
        ax.plot(m0,m0err,'bo',markersize=0.5)
        ax.plot(m0[p==1],m0err[p==1],'bo',markersize=5)
        ax.plot(m0[p==0],m0err[p==0],'rx',markersize=5,mew=2)
        ax.plot(m0[keys == 'Target'],m0err[keys == 'Target'],'ro',ms=5)
        ax.axis([np.min(m0)-np.abs(np.max(m0)-np.min(m0))*0.03,
                 np.max(m0)+np.abs(np.max(m0)-np.min(m0))*0.03,
                 np.min(m0err)-np.abs(np.max(m0err)-np.min(m0err))*0.03,
                 np.max(m0err)+np.abs(np.max(m0err)-np.min(m0err))*0.03])
        if np.min(n_valid) < n_min:
            ax.plot(m0[n_valid<n_min],m0err[n_valid<n_min],'o',ms=10,
                    fillstyle='none')
        #ax.plot(m0[keys=='Target'],target_err,'^',color='green')
        ax.axvline(m0[keys=='Target'],ls='--',color='green')

        #bx_m0,bx_m0err=boxcar(m0[p==1][np.argsort(m0[p==1])],m0err[p==1][np.argsort(m0[p==1])],width=bx_width)
        #ax.plot(bx_m0,bx_m0err,'-',color='green',markersize=5)
        
        #dm0 = np.min((m0[p==1][np.argsort(m0[p==1])] - np.roll(m0[p==1][np.argsort(m0[p==1])],1))[1:])
        x_new = np.linspace(np.min(m0[p==1]),np.max(m0[p==1]),1000)
        dm0 = x_new[1]-x_new[0]
        flinear = interpolate.interp1d(m0[p==1][np.argsort(m0[p==1])],m0err[p==1][np.argsort(m0[p==1])])
        y_linear = flinear(x_new)
        y_smooth = gaussian_filter(y_linear,s_width/dm0)
        ax.plot(x_new,y_smooth,'-',color='green',markersize=5)

        ax.set_xlabel('Mean Mag')
        ax.set_ylabel('Standard Deviation of Mean Mag')
        ax.set_title('Mean Magnitude Plot')
        plt.draw()
    # Function to make PLOT I

    def lc_plot(lc_hjd,lc_mag,lc_magerr,lc_p):
        ax.set_title('Lightcurve Plot - '+lc_id[0]+' - '+'N='+
                     str(np.sum(lc_p==1))+' - STD = '+str(np.round(np.std(lc_mag[lc_p==1]),4)))
        ax.plot(lc_hjd,lc_mag,'bo',ms=0.5)
        ax.plot(lc_hjd[lc_p==1],lc_mag[lc_p==1],'bo',markersize=5)
        ax.errorbar(lc_hjd,lc_mag,lc_magerr,fmt='o',ms=5)
        ax.plot(lc_hjd[lc_p==0],lc_mag[lc_p==0],'rx',markersize=8,mew=2)
        ax.plot([np.min(lc_hjd)-np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03,
                 np.max(lc_hjd)+np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03],
                [np.mean(lc_mag[lc_p==1]),np.mean(lc_mag[lc_p==1])],'--',
                color='black')
        ax.plot([np.min(lc_hjd)-np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03,
                 np.max(lc_hjd)+np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03],
                [np.mean(lc_mag[lc_p==1])+2.0*np.mean(lc_magerr[lc_p==1]),
                 np.mean(lc_mag[lc_p==1])+2.0*np.mean(lc_magerr[lc_p==1])],':',
                color='black')
        ax.plot([np.min(lc_hjd)-np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03,
                 np.max(lc_hjd)+np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03],
                [np.mean(lc_mag[lc_p==1])-2.0*np.mean(lc_magerr[lc_p==1]),
                 np.mean(lc_mag[lc_p==1])-2.0*np.mean(lc_magerr[lc_p==1])],':',
                color='black')
        ax.plot([np.min(lc_hjd)-np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03,
                 np.max(lc_hjd)+np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03],
                [np.mean(lc_mag[lc_p==1])+2.0*np.std(lc_mag[lc_p==1]),
                 np.mean(lc_mag[lc_p==1])+2.0*np.std(lc_mag[lc_p==1])],'--',
                color='black')
        ax.plot([np.min(lc_hjd)-np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03,
                 np.max(lc_hjd)+np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03],
                [np.mean(lc_mag[lc_p==1])-2.0*np.std(lc_mag[lc_p==1]),
                 np.mean(lc_mag[lc_p==1])-2.0*np.std(lc_mag[lc_p==1])],'--',
                color='black')
        xmin=(np.min(lc_hjd)-np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03)
        xmax=(np.max(lc_hjd)+np.abs(np.max(lc_hjd)-np.min(lc_hjd))*0.03)
        ymin=(np.max(lc_mag[lc_p==1]+lc_magerr[lc_p==1])+
              np.abs(np.max(lc_mag[lc_p==1]+lc_magerr[lc_p==1])-
                     np.min(lc_mag[lc_p==1]-lc_magerr[lc_p==1]))*0.03)
        ymax=(np.min(lc_mag[lc_p==1]-lc_magerr[lc_p==1])-
              np.abs(np.max(lc_mag[lc_p==1]+lc_magerr[lc_p==1])-
                     np.min(lc_mag[lc_p==1]-lc_magerr[lc_p==1]))*0.03)
        ax.axis([xmin,xmax,ymin,ymax])
        ax.set_xlabel('HJD (days)')
        ax.set_ylabel('Measured Mag')
        plt.draw()
    # Function to make PLOT II

    def em_plot(em,em_p):
        frame=np.arange(em.size)
        ax.set_title('Exposure Magnitude Plot')
        ax.plot(frame,em,'bo',ms=0.5)
        ax.plot(frame[em_p==1],em[em_p==1],'bo',ms=5)
        ax.plot(frame[em_p==0],em[em_p==0],'rx',ms=8,mew=2)
        ax.errorbar(range(em.size),em,emerr,fmt='o',ms=5)
        ax.set_xlabel('Exposure (Frame) Number')
        ax.set_ylabel('Exposure (Frame) Mag')
        ax.axis([np.min(frame)-np.abs(np.max(frame)-np.min(frame))*0.03,
                 np.max(frame)+np.abs(np.max(frame)-np.min(frame))*0.03,
                 np.min(em-emerr)-
                 np.abs(np.max(em-emerr)-np.min(em-emerr))*0.03,
                 np.max(em+emerr)+
                 np.abs(np.max(em-emerr)-np.min(em-emerr))*0.03])
        plt.draw()
    # Function to make PLOT III

    def press_key(event,plotted=plotted,p=p,lc_p=lc_p,lc_id=lc_id,em_p=em_p,
                  lc_hjd=lc_hjd,lc_mag=lc_mag,lc_magerr=lc_magerr):
        # PLOT I Key Commands - Mean Magnitude Plot

        if plotted[0] == 'mags':
            x_norm=np.abs(np.max(m0)-np.min(m0))*0.8
            y_norm=np.abs(np.max(m0err)-np.min(m0err))
            # Normalization takes into account axis scales and apsect ratio
            d=np.sqrt((event.xdata/x_norm-m0/x_norm)**2+
                      (event.ydata/y_norm-m0err/y_norm)**2)

            if event.key == 'd':
                if p[d == np.min(d)] == 0:
                    i=0
                    done='no'
                    while done != 'yes':
                        i=i+1
                        if p[d == (d[np.argsort(d)])[i]] == 0:
                            done = 'no'
                        if p[d == (d[np.argsort(d)])[i]] == 1:
                            p[d == (d[np.argsort(d)])[i]]=0
                            to_d=(d == (d[np.argsort(d)])[i])
                            for j in range(ee.size):
                                ((matches[(ss[to_d])[0]])[j])[13]=1.0*10**-20
                            done = 'yes'
                            print((ss[to_d])[0],': deleted')
                if p[d == np.min(d)] == 1:
                    p[d == np.min(d)]=0
                    to_d=(d == np.min(d))
                    for j in range(ee.size):
                        ((matches[(ss[to_d])[0]])[j])[13]=1.0*10**-20
                    print((ss[to_d])[0],': deleted')
                plt.cla()
                plot_mags(m0,m0err,p,n_valid,np.mean(ens_mag_err))

            if event.key == 'u':
                if 0 in p:
                    if p[d == np.min(d)] == 1:
                        i=0
                        done='no'
                        while done != 'yes':
                            i=i+1
                            if p[d == (d[np.argsort(d)])[i]] == 1:
                                done = 'no'
                            if p[d == (d[np.argsort(d)])[i]] == 0:
                                p[d == (d[np.argsort(d)])[i]]=1
                                to_d=(d == (d[np.argsort(d)])[i])
                                for j in range(ee.size):
                                    ((matches[(ss[to_d])[0]])[j])[13]=1.0
                                print((ss[to_d])[0],': undeleted')
                                done = 'yes'
                    if p[d == np.min(d)] == 0:
                        p[d == np.min(d)]=1
                        to_d=(d == np.min(d))
                        for j in range(ee.size):
                            ((matches[(ss[to_d])[0]])[j])[13]=1.0
                        print((ss[to_d])[0],': undeleted')
                    plt.cla()
                    plot_mags(m0,m0err,p,n_valid,np.mean(ens_mag_err))

            if event.key == 'e':
                llc_mag=np.array([row[5] for row in matches[(ss[d == np.min(d)])[0]]]-em) - mag_shift

                llc_magerr=np.zeros(llc_mag.size)
                for i in range(llc_magerr.size):
                    if ss_eff[i]>0.1:
                        llc_magerr[i]=np.sqrt(np.array(([row[6] for row in matches[(ss[d==np.min(d)])[0]]]))[i]**2+emerr[i]**2/ss_eff[i])

                llc_hjd=np.array([row[2] for row in matches[(ss[d == np.min(d)])[0]]])
                llc_p=np.array([row[14]*row[12] for row in matches[(ss[d==np.min(d)])[0]]],dtype=int)

                lc_id[0]=(ss[d == np.min(d)])[0]

                #You have to do the loop below because when you delete things, 
                #this makes sure you update all the arrays. 
                for i in range(llc_mag.size):
                    lc_mag[i] = llc_mag[i]
                    lc_magerr[i] = llc_magerr[i]
                    lc_hjd[i] = llc_hjd[i]
                    lc_p[i] = llc_p[i]
                plt.cla()
                lc_plot(lc_hjd,lc_mag,lc_magerr,lc_p)
                plotted[0]='lc'

            if event.key == 'w':
                em_p=np.array([((matches[ss[0]])[i])[12] for \
                               i in range(ee.size)],dtype=int)
                plt.cla()
                em_plot(em,em_p)
                plotted[0]='em'


        # PLOT III Key Commands - Exposure Magnitude Plot
        if plotted[0] == 'em':
            x_norm=np.abs(np.max(np.arange(em.size))-
                          np.min(np.arange(em.size)))*0.8
            y_norm=np.abs(np.max(em+emerr)-np.min(em-emerr))
            d=np.sqrt(((event.xdata/x_norm)-(np.arange(em.size)/x_norm))**2+
                      ((event.ydata/y_norm)-(em/y_norm))**2)

            if event.key == 'b':
                plotted[0]='mags'
                plt.cla()
                plot_mags(m0,m0err,p,n_valid,np.mean(ens_mag_err))

            if event.key == 'd':
                if em_p[d == np.min(d)] == 0:
                    i=0
                    done='no'
                    while done != 'yes':
                        i=i+1
                        if em_p[d == (d[np.argsort(d)])[i]] == 0:
                            done = 'no'
                        if em_p[d == (d[np.argsort(d)])[i]] == 1:
                            em_p[d == (d[np.argsort(d)])[i]]=0
                            to_d=np.where(d == (d[np.argsort(d)])[i])[0]
                            for j in range(ss.size):
                                ((matches[ss[j]])[to_d[0]])[12]=1.0*10**-20
                            ((matches['Target'])[to_d[0]])[12]=1.0*10**-20
                            print('Exp ',to_d[0],': deleted')
                            done='yes'
                if em_p[d == np.min(d)] == 1:
                    em_p[d == np.min(d)]=0
                    to_d=np.where(d == np.min(d))[0]
                    for j in range(ss.size):
                        ((matches[ss[j]])[to_d[0]])[12]=1.0*10**-20
                    ((matches['Target'])[to_d[0]])[12]=1.0*10**-20
                    print('Exp ',to_d[0],': deleted')
                plt.cla()
                em_plot(em,em_p)

            if event.key == 'u':
                if 0 in em_p: 
                    if em_p[d == np.min(d)] == 1:
                        i=0
                        done='no'
                        while done != 'yes':
                            i=i+1
                            if em_p[d == (d[np.argsort(d)])[i]] == 1:
                                done = 'no'
                            if em_p[d == (d[np.argsort(d)])[i]] == 0:
                                em_p[d == (d[np.argsort(d)])[i]]=1
                                to_d=np.where(d == (d[np.argsort(d)])[i])[0]
                                for j in range(ss.size):
                                    ((matches[ss[j]])[to_d[0]])[12]=1.0
                                ((matches['Target'])[to_d[0]])[12]=1.0
                                print('Exp ',to_d[0],': undeleted')
                                done='yes'
                    if em_p[d == np.min(d)] == 0:
                        em_p[d == np.min(d)]=1
                        to_d=np.where(d == np.min(d))[0]
                        for j in range(ss.size):
                            ((matches[ss[j]])[to_d[0]])[12]=1.0
                        ((matches['Target'])[to_d[0]])[12]=1.0
                        print('Exp ',to_d[0],': undeleted')
                    plt.cla()
                    em_plot(em,em_p)


        # PLOT II Key Commands - Lightcurve Plot
        if plotted[0] == 'lc':
            x_norm=np.abs(np.max(lc_hjd[lc_p==1])-np.min(lc_hjd[lc_p==1]))*0.8
            y_norm=np.abs(np.max(lc_mag[lc_p==1]+lc_magerr[lc_p==1])-
                          np.min(lc_mag[lc_p==1]-lc_magerr[lc_p==1]))
            d=np.sqrt((event.xdata/x_norm-lc_hjd/x_norm)**2+
                      (event.ydata/y_norm-lc_mag/y_norm)**2)

            if event.key == 'b':
                plotted[0]='mags'
                plt.cla()
                plot_mags(m0,m0err,p,n_valid,np.mean(ens_mag_err))

            if event.key == 'd':
                if lc_p[d == np.min(d)] == 0:
                    i=0
                    done='no'
                    while done != 'yes':
                        i=i+1
                        if lc_p[d == (d[np.argsort(d)])[i]] == 0:
                            done = 'no'
                        if lc_p[d == (d[np.argsort(d)])[i]] == 1:
                            lc_p[d == (d[np.argsort(d)])[i]]=0
                            to_d=np.where(d == (d[np.argsort(d)])[i])
                            ((matches[lc_id[0]])[(to_d[0])[0]])[14]=1.0*10**-20
                            print(lc_id[0],'-',(to_d[0])[0],': deleted')
                            done = 'yes'
                if lc_p[d == np.min(d)] == 1:
                    lc_p[d == np.min(d)]=0
                    to_d=np.where(d == np.min(d))
                    ((matches[lc_id[0]])[(to_d[0])[0]])[14]=1.0*10**-20
                    print(lc_id[0],'-',(to_d[0])[0],': deleted')
                plt.cla()
                lc_plot(lc_hjd,lc_mag,lc_magerr,lc_p)

            if event.key == 'u':
                if 0 in lc_p:
                    if lc_p[d == np.min(d)] == 1:
                        i=0
                        done='no'
                        while done != 'yes':
                            i=i+1
                            if lc_p[d == (d[np.argsort(d)])[i]] == 1:
                                done = 'no'
                            if lc_p[d == (d[np.argsort(d)])[i]] == 0:
                                lc_p[d == (d[np.argsort(d)])[i]]=1
                                to_d=np.where(d == (d[np.argsort(d)])[i])
                                ((matches[lc_id[0]])[(to_d[0])[0]])[14]=1.0
                                print(lc_id[0],'-',(to_d[0])[0],': undeleted')
                                done = 'yes'
                    if lc_p[d == np.min(d)] == 0:
                        lc_p[d == np.min(d)]=1
                        to_d=np.where(d == np.min(d))
                        ((matches[lc_id[0]])[(to_d[0])[0]])[14]=1.0
                        print(lc_id[0],'-',(to_d[0])[0],': undeleted')
                    plt.cla()
                    lc_plot(lc_hjd,lc_mag,lc_magerr,lc_p)

            #if event.key == 'o':
            #    mag_val_d=lc_mag[np.abs(lc_mag-np.mean(lc_mag[lc_p==1]))==
            #                     np.max(np.abs(lc_mag[lc_p==1]-
            #                                   np.mean(lc_mag[lc_p==1])))]
            #    to_d=np.where(lc_mag==mag_val_d)
            #    lc_p[to_d[0][0]]=0
            #    ((matches[lc_id[0]])[to_d[0][0]])[14]=1.0*10**-20
            #    print(lc_id[0],'-',to_d[0][0],': deleted')
            #    plt.cla()
            #    lc_plot(lc_hjd,lc_mag,lc_magerr,lc_p)

    print(' ')
    print('A plot of the mean magnitude vs mean magnitude error')
    print(' ')
    print('Key commands from Mean Magnitude Plot:')
    print('d - delete star from ensemble')
    print('u - undelete star from ensemble')
    print('e - inspect the lightcurve for that star')
    print('w - inspect ensemble exposure magnitudes')
    print(' ')
    print('Key commands for Lightcurve Plot:')
    print("d - delete point from that star's lightcurve")
    print("u - undelete point from that star's lightcurve")
    print('b - return to Mean Magnitude Plot')
    print(' ')
    print('Key commands for Exposure Magnitude Plot:')
    print('d - delete exposure from ensemble')
    print('u - undelete exposure from ensemble')
    print('b - return to Mean Magnitude Plot')
    print(' ')
    print('Hit return in the python terminal to quit')

    ss_eff=np.zeros(ee.size) #number of stars with non-zero (10^-20) weights. 
    for i in range(ee.size):
        k=0
        for j in range(ss.size):
            if (matches[ss[j]][i][12] == 1.0)&(matches[ss[j]][i][13] == 1.0)& \
               (matches[ss[j]][i][14] == 1.0) == True:
                k=k+1
        ss_eff[i]=k
    norm_emerr=np.sqrt((emerr[np.where(ss_eff > 0)])**2 / 
                       ss_eff[np.where(ss_eff > 0)])
    g=np.array([row[14]*row[12] for row in matches['Target']]) > 0.1
    t_mag=np.array([row[5] for row in matches['Target']])
    t_mag=t_mag[(g == True) & (ss_eff > 0)] - mag_shift
    t_magerr=np.array([row[6] for row in matches['Target']])
    t_magerr=t_magerr[(g == True) & (ss_eff > 0)]
    t_ss_eff=ss_eff[(g == True) & (ss_eff > 0)]
    t_em=em[(g == True) & (ss_eff > 0)]
    t_emerr=emerr[(g == True) & (ss_eff > 0)]
    ens_mag_err=np.sqrt(t_magerr**2+((t_emerr)**2 / t_ss_eff))

    n_valid=np.zeros(ss.size)
    for i in range(ss.size):
        n_valid[i]=np.sum(np.array([row[14]*row[12] for row in matches[(ss[i])]],dtype=int))

    print('')
    print('Mean exposure error: ',np.mean(ens_mag_err))
    print('Edit comparison stars? Yes (y), No (n)?')
    redo=input('-->')

    if redo not in ('y','n'):
        while redo not in ('y','n'):
            print('Try Again')
            redo = input('-->')
            
    if redo == 'y':
        while redo == 'y':
            keys,ee,ss,a,b,mag_es,w_es,x,em,emerr,m0,m0err=ens_fill(matches)

            p=np.array([((matches[ss[i]])[0])[13] for i in range(ss.size)],dtype=int)

            mag_shift = np.min(m0[p==1])-1
            m0 = m0 - mag_shift

            fig=plt.figure()
            ax=fig.add_subplot(111)
            plot_mags(m0,m0err,p,n_valid,np.mean(ens_mag_err))
            plotted[0]='mags'
            cid = fig.canvas.mpl_connect('key_press_event', press_key)

            wait=input('')

            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            plotted[0]='mags'

            keys,ee,ss,a,b,mag_es,w_es,x,em,emerr,m0,m0err=ens_fill(matches)

            mag_shift = np.min(m0[p==1])-1
            m0 = m0 - np.min(m0[p==1])+1

            ss_eff=np.zeros(ee.size) #num of non-zero (10^-8) weight stars. 
            for i in range(ee.size):
                k=0
                for j in range(ss.size):
                    if (matches[ss[j]][i][12] == 1.0) &\
                       (matches[ss[j]][i][13] == 1.0) &\
                       (matches[ss[j]][i][14] == 1.0) == True:
                        k=k+1
                ss_eff[i]=k
            norm_emerr=np.sqrt((emerr[np.where(ss_eff > 0)])**2 / 
                               ss_eff[np.where(ss_eff > 0)])
            g=np.array([row[14]*row[12] for row in matches['Target']]) > 0.1
            t_mag=np.array([row[5] for row in matches['Target']])
            t_mag=t_mag[(g == True) & (ss_eff > 0)] - mag_shift
            t_magerr=np.array([row[6] for row in matches['Target']])
            t_magerr=t_magerr[(g == True) & (ss_eff > 0)]
            t_ss_eff=ss_eff[(g == True) & (ss_eff > 0)]
            t_em=em[(g == True) & (ss_eff > 0)]
            t_emerr=emerr[(g == True) & (ss_eff > 0)]
            ens_mag_err=np.sqrt(t_magerr**2+((t_emerr)**2 / t_ss_eff))

            #print('adjusting error...')
            #dm0 = np.min((m0[p==1][np.argsort(m0[p==1])] - np.roll(m0[p==1][np.argsort(m0[p==1])],1))[1:])
            #x_new = np.arange(np.min(m0[p==1]),np.max(m0[p==1]),dm0)
            #flinear = interpolate.interp1d(m0[p==1][np.argsort(m0[p==1])],m0err[p==1][np.argsort(m0[p==1])])
            #y_linear = flinear(x_new)
            #y_smooth = gaussian_filter(y_linear,s_width/dm0)
            #emp_std = y_smooth[np.abs(x_new-m0[keys=='Target']) == np.min(np.abs(x_new-m0[keys=='Target']))]
            #
            #err_inflate = np.sqrt(emp_std**2 - np.min(ens_mag_err)**2)
            #
            #ens_mag_err = np.sqrt(ens_mag_err**2 + err_inflate**2)

            n_valid=np.zeros(ss.size)
            for i in range(ss.size):
                n_valid[i]=np.sum(np.array([row[14]*row[12] for row in 
                                             matches[(ss[i])]],dtype=int))
            print('')
            print('Mean exposure error: ',np.mean(ens_mag_err))

            print('Re-compute magnitudes? Yes (y), No (n)?')
            redo=input('-->')
            if redo not in ('y','n'):
                while redo not in ('y','n'):
                    print('Try Again')
                    redo = input('-->')

    if redo == 'n':
        keys,ee,ss,a,b,mag_es,w_es,x,em,emerr,m0,m0err=ens_fill(matches)

        p=np.array([((matches[ss[i]])[0])[13] for i in range(ss.size)], dtype=int)

        mag_shift = np.min(m0[p==1])-1
        m0 = m0 - mag_shift

        g=np.array([row[14]*row[12] for row in matches['Target']]) > 0.1
        # Select only good matches and good exposures.

        t_hjd=np.array([row[2] for row in matches['Target']])
        t_hjd=t_hjd[(g == True) & (ss_eff > 0)]

        t_air=np.array([row[16] for row in matches['Target']])
        t_air=t_air[(g == True) & (ss_eff > 0)]

        t_mag=np.array([row[5] for row in matches['Target']])
        t_mag=t_mag[(g == True) & (ss_eff > 0)] - mag_shift

        t_magerr=np.array([row[6] for row in matches['Target']])
        t_magerr=t_magerr[(g == True) & (ss_eff > 0)]

        t_ss_eff=ss_eff[(g == True) & (ss_eff > 0)]

        t_em=em[(g == True) & (ss_eff > 0)]

        t_emerr=emerr[(g == True) & (ss_eff > 0)]

        ens_mag=(t_mag-t_em)
        ens_mag_err=np.sqrt(t_magerr**2+((t_emerr)**2 / t_ss_eff))

        print('adjusting error...')
        dm0 = np.min((m0[p==1][np.argsort(m0[p==1])] - np.roll(m0[p==1][np.argsort(m0[p==1])],1))[1:])
        x_new = np.arange(np.min(m0[p==1]),np.max(m0[p==1]),dm0)
        flinear = interpolate.interp1d(m0[p==1][np.argsort(m0[p==1])],m0err[p==1][np.argsort(m0[p==1])])
        y_linear = flinear(x_new)
        y_smooth = gaussian_filter(y_linear,s_width/dm0)
        emp_std = y_smooth[np.abs(x_new-m0[keys=='Target']) == np.min(np.abs(x_new-m0[keys=='Target']))]

        err_inflate = np.sqrt(emp_std**2 - np.min(ens_mag_err)**2)

        ens_mag_err = np.sqrt(ens_mag_err**2 + err_inflate**2)

        t_ra_i=np.array([row[10] for row in matches['Target']])
        t_ra=np.mean(t_ra_i[(g == True) & (ss_eff > 0)])
        t_ra_std=np.std(t_ra_i[(g == True) & (ss_eff > 0)])

        t_dec_i=np.array([row[11] for row in matches['Target']])
        t_dec=np.mean(t_dec_i[(g == True) & (ss_eff > 0)])
        t_dec_std=np.std(t_dec_i[(g == True) & (ss_eff > 0)])

        t_ss=np.array([0],dtype='int')

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(t_hjd,ens_mag,'bo',markersize=5)
        ax.errorbar(t_hjd,ens_mag,ens_mag_err,fmt='o',markersize=5)
        ax.set_xlabel('HJD')
        ax.set_ylabel('$\Delta$ Mag')
        ax.set_title('Ensemble Target Lightcurve')
        ax.axis([np.min(t_hjd)-np.abs(np.max(t_hjd)-np.min(t_hjd))*0.03,
                 np.max(t_hjd)+np.abs(np.max(t_hjd)-np.min(t_hjd))*0.03,
                 np.max(ens_mag+ens_mag_err)+
                 np.abs(np.max(ens_mag+ens_mag_err)-
                        np.min(ens_mag-ens_mag_err))*0.03,
                 np.min(ens_mag-ens_mag_err)-
                 np.abs(np.max(ens_mag+ens_mag_err)-
                        np.min(ens_mag-ens_mag_err))*0.03])
        wait=input('')
        plt.close(fig)
        # Plot target lightcurve
        print('Output as data text files? Yes (y), No (n).')
        output = input('-->')
        if output not in ('y','n'):
            while output not in ('y','n'):
                print('Try Again')
                output = input('-->')


        if output == 'y':
            np.savetxt(str(target)+'_'+str(date)+'_'+str(((matches['Target'])[0])[7])+'_enslc_'+str(scope)+'.dat',
                       np.c_[t_hjd,ens_mag,ens_mag_err,t_ss_eff,t_air],
                       fmt=['%15.10f','%10.5f','%10.5f','%4i','%10.5f'])

            if not os.path.exists('ens_data'):
                os.makedirs('ens_data')
            if not os.path.exists('./ens_data/'+date):
                os.makedirs('./ens_data/'+date)

            pickle.dump(matches,open(in_file[:-2]+'_ens_final.p','wb'))
            os.system('mv '+in_file[:-2]+'_ens_final.p ./ens_data/'+date)

            pp = PdfPages(str(target)+'_'+str(date)+'_'+str(((matches['Target'])[0])[7])+'_enshist_'+str(scope)+'.pdf')
            ax = plt.subplot(111)
            ax.set_title('Matched Stars')
            ax.set_ylabel('# of Images')
            ax.set_xlabel('# of Comparison Stars Used')
            ax.hist(ss_eff,bins=np.sort(np.append([np.unique(ss_eff)-0.5],
                                                  [np.unique(ss_eff)+0.5])))
            pp.savefig()
            pp.close()
            plt.close()
            os.system('mv '+str(target)+'_'+str(date)+'_'+str(((matches['Target'])[0])[7])+'_enshist_'+str(scope)+'.pdf ./ens_data/'+str(date))

            c_ra=np.empty(len(ss[keys != 'Target']),dtype='float64')
            c_ra_std=np.empty(len(ss[keys != 'Target']),dtype='float64')
            c_dec=np.empty(len(ss[keys != 'Target']),dtype='float64')
            c_dec_std=np.empty(len(ss[keys != 'Target']),dtype='float64')
            
            comps=(keys!='Target')

            for i in range(len(ss[comps])):
                ra_i=np.array([row[10] for row in matches[ss[comps][i]]],dtype='float64')
                ra_g=np.array([row[14] for row in matches[ss[comps][i]]]) > 0.1

                c_ra[i]=np.mean(ra_i[ra_g])
                
                c_ra_std[i]=np.std(ra_i[ra_g])
                
                dec_i=np.array([row[11] for row in matches[ss[comps][i]]],dtype='float64')
                dec_g=np.array([row[14] for row in matches[ss[comps][i]]]) > 0.1

                c_dec[i]=np.mean(dec_i[dec_g])
                c_dec_std[i]=np.std(dec_i[dec_g])

            ss_mag_mean=np.empty(0)
            ss_mag_err_mean=np.empty(0)
            ss_mag_std=np.empty(0)

            for i in range(len(ss[comps])):
                g=np.array([row[14]*row[12] for row in matches[ss[comps][i]]])>0.1
                # Select only good matches and good exposures.
                
                ss_hjd=np.array([row[2] for row in matches[ss[comps][i]]])
                ss_hjd=ss_hjd[(g == True) & (ss_eff > 0)]

                ss_air=np.array([row[16] for row in matches[ss[comps][i]]])
                ss_air=ss_air[(g == True) & (ss_eff > 0)]

                ss_mag=np.array([row[5] for row in matches[ss[comps][i]]])
                ss_mag=ss_mag[(g == True) & (ss_eff > 0)] - mag_shift

                ss_magerr=np.array([row[6] for row in matches[ss[comps][i]]])
                ss_magerr=ss_magerr[(g == True) & (ss_eff > 0)]

                ss_ss_eff=ss_eff[(g == True) & (ss_eff > 0)]

                ss_em=em[(g == True) & (ss_eff > 0)]

                ss_emerr=emerr[(g == True) & (ss_eff > 0)]

                ss_ens_mag=(ss_mag-ss_em)

                ss_ens_mag_err=np.sqrt(ss_magerr**2+((ss_emerr)**2 / ss_ss_eff))
                emp_std = y_smooth[np.abs(x_new-m0[keys==ss[comps][i]]) == np.min(np.abs(x_new-m0[keys==ss[comps][i]]))]

                err_inflate = np.sqrt(emp_std**2 - np.min(ss_ens_mag_err)**2)

                ss_ens_mag_err = np.sqrt(ss_ens_mag_err**2 + err_inflate**2)

                ss_mag_mean=np.append(ss_mag_mean,np.mean(ss_ens_mag))

                ss_mag_err_mean=np.append(ss_mag_err_mean, np.mean(ss_ens_mag_err))

                ss_mag_std=np.append(ss_mag_std,np.std(ss_ens_mag))
                    
                np.savetxt(str(ss[comps][i])+'_'+str(date)+'_'+str(((matches['Target'])[0])[7])+'_enslc_'+str(scope)+'.dat',
                           np.c_[ss_hjd,ss_ens_mag,ss_ens_mag_err,ss_ss_eff,ss_air],
                           fmt=['%15.10f','%10.5f','%10.5f','%4i','%10.5f'])

                os.system('mv '+str(ss[comps][i])+'_'+str(date)+'_'+
                          str(((matches['Target'])[0])[7])+'_enslc_'+str(scope)+
                          '.dat ./ens_data/'+str(date))

            f=open(str(target)+'_'+str(date)+'_'+str(((matches['Target'])[0])[7])+\
                       '_enscomps_'+str(scope)+'.dat','w')
            f.write('# ID    Mag0    Merr0   RA    '+
                    '          RA STD          DEC             DEC STD       '+
                    '  Mag     Merr    Magstd  \n')
            f.write(str(t_ss[0]).zfill(4)+'\t'+
                    "%5.3f" % m0[keys=='Target']+'\t'+
                    "%5.3f" % m0err[keys=='Target']+'\t'+
                    "%9.5f" % t_ra+'\t'+
                    "%g" % t_ra_std+'\t'+
                    "%9.5f" % t_dec+'\t'+
                    "%g" % t_dec_std+'\t'+
                    "%5.3f" % np.mean(ens_mag)+'\t'+
                    "%5.3f" % np.mean(ens_mag_err)+'\t'+
                    "%5.3f" % np.std(ens_mag)+'\n')

            for i in range(ss[(p==1) & (keys!='Target')].size):
                f.write(str(ss[(p==1) & (keys!='Target')][i]).zfill(4)+'\t'+
                        "%5.3f" % m0[(p==1) & (keys!='Target')][i]+'\t'+
                        "%5.3f" % m0err[(p==1) & (keys!='Target')][i]+'\t'+
                        "%9.5f" % c_ra[p[comps]==1][i]+'\t'+
                        "%g" % c_ra_std[p[comps]==1][i]+'\t'+
                        "%9.5f" % c_dec[p[comps]==1][i]+'\t'+
                        "%g" % c_dec_std[p[comps]==1][i]+'\t'+
                        "%5.3f" % ss_mag_mean[(p[keys!='Target']==1)][i]+'\t'+
                        "%5.3f" % ss_mag_err_mean[(p[keys!='Target']==1)][i]+
                        '\t'+
                        "%5.3f" % ss_mag_std[(p[keys!='Target']==1)][i]+'\n')
            f.close()

            os.system('mv '+str(target)+'_'+str(date)+'_'+str(((matches['Target'])[0])[7])+\
                          '_enscomps_'+str(scope)+'.dat ./ens_data/'+str(date))

    return t_hjd,ens_mag,ens_mag_err


def ens_fill(matches):

    '''A program that creates and solves the system of linear equations 
    outlined in Honeycutt 1992 (i.e. populating matricies). Reads in the
    pyred.ens_match output dictionary. The system of equations are setup to 
    minimiize the standard deviation of the mean magnitude of each star 
    simultaneously. The solution provides the exposure magnitudes for each 
    frame and the mean magnitude for each star.

    Parameters:
    -----------
    matches: dictionary
        Python dictionary output by pyred.ens_match().

    Returns:
    --------
    keys : ndarray; str
        Dictionary calls for the Target and all comparison stars in the 
        matched dictionary.

    ee : ndarray; str
        File names for each exposure in ensemble.

    ss : ndarray; str
        Dictionary calls for all comparison stars in the ensemble.
    
    a : ndarray
        ee+ss square matrix formed form the system of linear equations outlined 
        in Honeycutt 1992 in the ax=b formalism.
        (This may not actually need to be returned, was probably just for 
        testing.)

    b : ndarray
        1 by ee+ss array formed form the system of linear equations outlined 
        in Honeycutt 1992 in the ax=b formalism.
        (This may not actually need to be returned, was probably just for 
        testing.)

    mag_es : ndarray 
        Array of instrumental magnitudes for each star in each frame. 
        (This may not actually need to be returned, was probably just for 
        testing.)

    w_es : ndarray
        Array of each star's weight in a given exposure.
        (This may not actually need to be returned, was probably just for 
        testing.)

    x : ndarray
        1 by ee+ss array formed form the solution of system of linear 
        equations outlined in Honeycutt 1992 in the ax=b formalism. Array 
        contains the exposure mangnitudes and mean magnitudes. 
        (This may not actually need to be returned, was probably just for 
        testing.)

    em : ndarray
        Array of exposure magnitudes pulled out of x.

    emerr : ndarray
        Array of exposure magnitude errors.
    
    m0 : ndarray
        Array of mean magnitudes of each comparison star.

    m0err : ndarray
        Array of mean magnitude errors for each comparison star. 

    Output:
    -------
    N/A

    Version History:
    ----------------
    2015-05-15 - Start
    2015-11-02 - Now all stars are used for ensemble magnitude calculation, 
                 including the target. Will not affect results previous to this
                 change because the target would be deleted while running 
                 ens_lc anyway. 
                 Change made to make the program more general when you want 
                 photometry for the entire field without a specific target in 
                 mind.

    '''

    keys=np.array(list(matches.keys()))
    ee=np.array([row[0] for row in matches[keys[0]]])
    ss=np.array(keys,dtype=str)
    a=np.zeros((ee.size+ss.size,ee.size+ss.size))
    b=np.zeros((ee.size+ss.size))
    mag_es=np.zeros((ee.size,ss.size))
    w_es=np.zeros((ee.size,ss.size))

    for i in range(ee.size):
        w1=np.array([((matches[d])[i])[12] for d in ss])
        w2=np.array([((matches[d])[i])[13] for d in ss])
        w3=np.array([((matches[d])[i])[14] for d in ss])
        w4=np.array([((matches[d])[i])[15] for d in ss])
        mag=np.array([((matches[d])[i])[5] for d in ss])
        a[i,i]=np.sum(w1*w2*w3*w4)
        b[i]=np.sum(w1*w2*w3*w4*mag)
        
    for i in range(ee.size):
        for j in range(ss.size):
            w1=((matches[ss[j]])[i])[12]
            w2=((matches[ss[j]])[i])[13]
            w3=((matches[ss[j]])[i])[14]
            w4=((matches[ss[j]])[i])[15]
            a[i,j+ee.size]=w1*w2*w3*w4
            a[j+ee.size,i]=w1*w2*w3*w4
            mag_es[i,j]=((matches[ss[j]])[i])[5]
            w_es[i,j]=w1*w2*w3*w4

    for i in range(ss.size):
        w1=np.array([row[12] for row in matches[ss[i]]])
        w2=np.array([row[13] for row in matches[ss[i]]])
        w3=np.array([row[14] for row in matches[ss[i]]])
        w4=np.array([row[15] for row in matches[ss[i]]])
        mag=np.array([row[5] for row in matches[ss[i]]])
        a[i+ee.size,i+ee.size]=np.sum(w1*w2*w3*w4)
        b[i+ee.size]=np.sum(w1*w2*w3*w4*mag)

    x=np.linalg.solve(a,b)

    em=x[0:ee.size]
    m0=x[ee.size:]

    emerr=np.sqrt(ss.size*np.sum([(mag_es[:,d]-em-m0[d])**2 * w_es[:,d] 
                                  for d in range(ss.size)],axis=0) / 
                  ((ss.size-1.0)*np.sum(w_es,axis=1)))

    m0err=np.sqrt(ee.size*np.sum([(mag_es[d,:]-em[d]-m0)**2 * w_es[d,:] 
                                  for d in range(ee.size)],axis=0) /
                  ((ee.size-1.0)*np.sum(w_es,axis=0)))

    return keys,ee,ss,a,b,mag_es,w_es,x,em,emerr,m0,m0err


def boxcar(time,mag,width=None,symmetric=False,trim=False):
    '''
    Boxcar median smoothing of the input time-series data. You have the option 
    to specify the width of the smoothing kernel and whether the data set is 
    symmetric about its boundaries. 

    Parameters
    ----------
    time : array like
       An array of the times of observations. Units can be anything. Could be
       phase values, for instance. 

    mag : array like
       An array of the time dependent variable. Listed as magnitude but can be 
       anything. 

    width : float, optional
       The boxcar smoothing width in units of the time variable. If not set by
       the user, the default value is one-quarter the length of the data set. 

    symmetric : boolian, optional.
       The option to use symmetic boundary conditions when boxcar smoothing, 
       i.e. does the first point include averaging from the data at the end of
       an array. This is useful for boxcar smoothing phase-folded data. 
       By default the value is set to False, not using symmetric boundary
       conditions. Setting symmetric to True enables this feature. 

    trim : Boolian, optional
       The option to trim the output array so that every output point is 
       smoothed by the full boxcar window. This will make the length of the
       input and output arrays different. The default value is Flase, meaning 
       the program will not do this. Setting trim to True will only work if 
       symmetric is set to False. 

    Returns
    -------
    Two Arrays:
       The the x-value array, time and the boxcar smoothed function calculated 
       at each point in output time array. If trim is set to false, the input
       and output time arrays will be the same. 

    Outputs
    -------
    None

    Version History
    ---------------
    2016-12-06 - Start
    '''
    if width == None:
        window=(time[-1]-time[0])/4.0
    else:
        window=width

    s_mag=np.zeros(mag.size)

    if symmetric == False:
        for i in range(s_mag.size):
            if (time[i]-time[0]) < window/2.0:
                s_points=((time < time[i]+window/2.0))

            if ((time[i]-time[0] >= window/2.0)&
                (time[-1]-time[i] >= window/2.0)):
                s_points=((time > time[i]-window/2.0)&
                          (time < time[i]+window/2.0))

            if (time[-1]-time[i]) < window/2.0:
                s_points=((time > time[i]-window/2.0))

            s_mag[i]=np.median(mag[s_points])

    if (symmetric == False) & (trim == True):
        good=np.zeros(mag.size)
        for i in range(s_mag.size):
            if ((time[i]-time[0] >= window/2.0)&
                (time[-1]-time[i] >= window/2.0)):
                s_points((time > time[i]-window/2.0)&
                          (time < time[i]+window/2.0))
                good[i]=1.0
            s_mag[i]=np.median(mag[s_points])
        time = time[good==1.0]
        s_mag = s_mag[good==1.0]

    if symmetric == True:
        for i in range(s_mag.size):
            if (time[i]-time[0]) < window/2.0:
                s_points=((time < (time[i]+window/2.0)) | 
                          (time > (time[-1]-(window/2.0)+(time[i]-time[0]))))

            if ((time[i]-time[0] >= window/2.0) &
                (time[-1]-time[i] >= 1.0-window/2.0)):
                s_points=((time >= (time[i]-window/2.0))&
                          (time <= (time[i]+window/2.0)))

            if (time[-1]-time[i]) < window/2.0:
                s_points=((time > time[i]-window/2.0) | 
                          (time < time[0]+window/2.0-(time[-1]-time[i])))
                
            s_mag[i]=np.median(mag[s_points])

    return time,s_mag





