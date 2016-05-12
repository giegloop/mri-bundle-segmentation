from PIL import Image
import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv


from functions import *

#IMPORT MRI DATA CODE, but work with arrays!
MRI= os.path.join(data_path,'C:\\Users\\Antonio Orvieto\\Desktop\\SLT PROJECT\\data\\diff_data.nii.gz')
bvecs = np.genfromtxt('C:\\Users\\Antonio Orvieto\\Desktop\\SLT PROJECT\\data\\bvecs',dtype='str')
img = nib.load(MRI)
MRI_data = img.get_data()

#initialization of directions
bvecs_x = [float(i) for i in bvecs[:,0]]
bvecs_y = [float(i) for i in bvecs[:,1]]
bvecs_z = [float(i) for i in bvecs[:,2]]
bvecs=np.column_stack((bvecs_x,bvecs_y,bvecs_z))

#gets indexes of nonzero rows
isValid=np.zeros((len(bvecs), 1))
for i in range(len(bvecs_x)):
    isValid[i] = bvecs_x[i]>0.1 or bvecs_y[i]>0.1 or bvecs_z[i]>0.1

#for a first clustering I will use a section at x=60
#direction of diffusion may not work good... maybe I still need to
#work with the complete star.

#EMBEDDING 1
x=60;

#Calculate maximum diffusion for every point 

#diffusions = np.zeros((130,210,3))
#for i in range(len(MRI_data[0,0,:,0])):
#    for j in range(len(MRI_data[0,:,0,0])):
#        #extract the star representation as the minimum nonzero signal
#        found = False;        
#        while not(found):        
#            m = np.argmin(MRI_data[x,j,i,:])
#            if isValid[m]:
#                found=True
#        diffusions[i,j,:]= ....
#        print(i)

#CLUSTERING...
        
          
#OUTPUT
#img = Image.fromarray(data, 'RGB')
#img.save('my.png')