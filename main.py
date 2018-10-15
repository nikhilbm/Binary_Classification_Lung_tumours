import math as m
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.feature import greycomatrix, greycoprops
import math
from numpy import ndarray
import xlsxwriter

data=np.genfromtxt('stage1_labels.csv',delimiter=',')
data=data[1:len(data),1]

# Some constants 
INPUT_FOLDER = '../Dicom Folder/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

# Load the scans in given folder path
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
f=np.zeros((5,9))
for z in range(1,5):
    first_patient = load_scan(INPUT_FOLDER + patients[z])
    first_patient_pixels = get_pixels_hu(first_patient)
    im = Image.fromarray((first_patient_pixels[np.shape(first_patient_pixels)[0]/2]+1024)/8)
    #im.show()
    im = im.convert(mode="L")
    im = im.filter(ImageFilter.MedianFilter(15))
    #im.show()
    pxl=np.array(im)
    thresh = threshold_otsu(pxl)
    pxl2=255*(pxl>thresh)
    #im2 = Image.fromarray(pxl2)
    #im2.show()
    selem=square(150)
    closed = closing(pxl2, selem)
    #im2 = Image.fromarray(closed)
    #im2.show()
    a= closed-pxl2
    selem=square(20)
    closed = closing(a, selem)
    selem = disk(6)
    #closed = erosion(closed, selem)
    #im2 = Image.fromarray(closed)
    #im2.show()
    mul=(closed>150)*pxl
    mask=(255-closed)+mul
    mul=255*(mul>.44*(np.max(mul)-np.min(mask)))
    #mul = 255*(mul>thresh)
    mul = opening(mul, selem)
    #im2 = Image.fromarray(mul)
    #im2.show()

    im=mul/255
    count=1
    l=np.zeros((512,512))
    for i in range(1,511):   
        for j in range(1,511):
            if im[i,j] == 1:
                   if im[i,j-1] == 1:
                         if im[i-1,j+1] == 1 and im[i-1,j] == 0:
                               temp=l[i-1,j+1]
                               for k in range(i+1):   
                                    for m in range(1,511):
                                        if l[k,m] == temp:
                                              l[k,m]=l[i,j-1]
                         l[i,j]=l[i,j-1]

                   elif im[i-1,j-1] == 1:
                         if im[i-1,j+1] == 1 and im[i-1,j] == 0:
                               temp=l[i-1,j+1]
                               for k in range(i+1):   
                                    for m in range(1,511):
                                        if l[k,m] == temp:
                                              l[k,m]=l[i-1,j-1]
                         l[i,j]=l[i-1,j-1]
                     
                   elif im[i-1,j] == 1:
                         l[i,j]=l[i-1,j]
                   elif im[i-1,j+1] == 1:
                         l[i,j]=l[i-1,j+1]
                   else:
                         l[i,j]=count
                         count+=1
    #print count
    area=np.zeros(count)
    for i in range(1,511):   
        for j in range(1,511):
            area[l[i,j]]+=1
            
    area[0]=0
    area=area*(area>80)
    
    peri=np.zeros(count)
    for k in range(count):
        if area[k]!=0:
            for i in range(1,511):   
                for j in range(1,511):
                    if l[i,j]==k:
                        if l[i-1,j]==0 or l[i-1,j-1]==0 or l[i-1,j+1]==0 or l[i,j-1]==0 or l[i,j+1]==0 or l[i+1,j]==0 or l[i+1,j-1]==0 or l[i+1,j+1]==0:
                            peri[k]+=1

    rou=np.zeros(count)
    for i in range(count):
        if area[i] != 0:
            rou[i]=4*3.14*area[i]/(peri[i]*peri[i])
    mr=np.argmin(abs(1-rou))
    im=(l==mr)*1
    #im2 = Image.fromarray(im*255)
    #im2.show()
    centroidi=0
    centroidj=0
    count=0
    for i in range(1,511):   
        for j in range(1,511):
            if im[i,j]==1:
                centroidi+=i
                centroidj+=j
                count+=1
    centroidi=centroidi/count
    centroidj=centroidj/count
    area=area[mr]
    peri=peri[mr]

    im=im*pxl
    glcm = greycomatrix(im[centroidi-3:centroidi+3,centroidj-3:centroidj+3], [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256, symmetric=False, normed=True)
    rou=rou[mr]

    #patients[z]
    contrast=greycoprops(glcm, 'contrast')[0, 0]    #rou
    corr=greycoprops(glcm, 'correlation')[0, 0]     #area
    energy=greycoprops(glcm, 'energy')[0, 0]        #peri
    homo=greycoprops(glcm, 'homogeneity')[0, 0]     #centroidi #centroidj
    f[z,:]=[area ,peri ,rou ,centroidi ,centroidj ,contrast ,corr ,energy ,homo]
        
    
