#!/usr/bin/env python
# coding: utf-8

# # FITS data visualization and statistics with Astropy

# `author` Julien Morin <julien.morin@umontpellier.fr>
# 
# `date` 12 Oct 2022
# 
# Reference: __[https://docs.astropy.org/en/stable/io/fits/index.html](https://docs.astropy.org/en/stable/io/fits/index.html)__
# * load a FITS image w/ astropy.io.fits, display data and header
# * plot image and adjust
# * compute basic statistics

# ## Load data and retrieve information on extensions and header

# In[ ]:


# import and settings
get_ipython().run_line_magic('matplotlib', 'notebook')
#
import numpy as np
import matplotlib.pyplot as plt
#
import astropy.io.fits as pf
from astropy.visualization import astropy_mpl_style
from astropy.visualization import simple_norm
from astropy.visualization import MinMaxInterval
from astropy.visualization import hist


# In[ ]:


# load FITS image
hdul = pf.open('IC434R-001.fit')

# display the HDUs contained in IC434R-001.fit
hdul.info()

### ToDo: 
###  check the length and type of hdul


# In[ ]:


# ToDo: load NICMOSn4hk12010_mos.fits and check content


# In[ ]:


# load FITS image and print header
hdul = pf.open('IC434R-001.fit')
print(hdul[0].header)


# In[ ]:


# delete variable hdul
del hdul


# In[ ]:


# alternatively: get header without explicitly loading the HDU in a variable beforehand
pf.getheader('IC434R-001.fit')


# In[ ]:


# load FITS image
hdul = pf.open('IC434R-001.fit')
# retrieve the value and comment of a specific keyword
print(hdul[0].header['EXPTIME'])
print(hdul[0].header.comments['EXPTIME'])


# In[ ]:


# list available keywords
print(list(hdul[0].header.keys()))


# ## Load and display a FITS image

# In[ ]:


# load FITS image
hdul = pf.open('IC434R-001.fit')
# store the data in a variable
data = hdul[0].data # call by position
#data = hdul['PRIMARY'].data # call by label


# In[ ]:


# delete variable hdul
del hdul


# In[ ]:


# alternatively: get the data without explicitly loading the HDU in a variable beforehand
data = pf.getdata('IC434R-001.fit')


# In[ ]:


### ToDo: check data type and dimensions


# In[ ]:


# plot w/ imshow
plt.figure()
im = plt.imshow(data, origin='lower') # option origin='lower' results in same orientation as ds9
plt.show()


# In[ ]:


# optimize plot parameters for image visualization
plt.figure()
plt.style.use(astropy_mpl_style)
im = plt.imshow(data, origin='lower') # option origin='lower' results in same orientation as ds9
plt.show()


# In[ ]:


# Apply a square root scale to data visualization
plt.figure()
norm = simple_norm(data, 'sqrt')
plt.imshow(data, origin='lower', norm=norm)
plt.show()


# In[ ]:


### ToDo: 
###  experiment w/ other scales: 'linear', 'sqrt', 'power', 'log', 'asinh'
###  set levels w/ parameters min_cut/max_cut or min_percent/max_percent


# Check __[simple_norm documentation](https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html#astropy.visualization.mpl_normalize.simple_norm)__

# In[ ]:


interval = MinMaxInterval()
z1, z2 = interval.get_limits(data)
print(z1, z2)
plt.figure()
im = plt.imshow(data, origin='lower', vmin=z1, vmax=z2)
plt.show()


# In[ ]:


### ToDo: experiment w/ other available options: 
###  ManualInterval, PercentileInterval, AsymmetricPercentileInterval, and ZScaleInterval 


# For more details see __[astropy.visualization documentation](https://docs.astropy.org/en/stable/visualization/normalization.html)__ 

# ## Compute and display statistics of a FITS image

# In[ ]:


# load FITS image
hdul = pf.open('IC434R-001.fit')
# store the data in a variable
data = hdul[0].data
# plot data histogram w/ 100 bins using matplotlib's histogram
plt.figure()
plt.hist(data.ravel(), bins=100, log=True) # data.ravel() converts the image into a 1-D array
plt.show()
### ToDo: find the "best" number of bins


# In[ ]:


plt.figure()
hist(data.ravel(), bins='scott', log=True) # options 'scott' and 'freedman' provide automatic determination of #bins
plt.show()


# In[ ]:




