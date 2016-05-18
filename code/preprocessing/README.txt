SLT PROJECT -PREPROCESSING AND Jij COMPUTATION

Team members: Sjoerd van Bekhoven, Torres Garcia Moises, Antonio Orvieto


1) Set correct paths in the MATLAB and the Python programs, add ellipsoid_fit to path.

2) Run the MATLAB file, pray.

3) Add the following lines in python with correct paths to import the compressed data,
   python does not actually need to import the full nii file, just these two text files.
   
   max_diff = np.genfromtxt('C:\\Users\\Antonio Orvieto\\Desktop\\preprocessing\\embeddings',dtype='str')
   FA = np.genfromtxt('C:\\Users\\Antonio Orvieto\\Desktop\\preprocessing\\embeddings',dtype='str')
   
   the data relative to the 3d point (i,j,k) is found in the row i*j*k of the matrices(they are generated 
   writing sequential lines in nested for loops, but you can reshape as you want) 
   
4) FA(210(i-1)+210(j-1)+k) (anisotropy) gives you an important information : is the 3d point white matter or not.
   Not all data points need to be considered in the clustering, only white matter. to do this you need to set
   a threshold. To get a nice result pick only data point with F>0.34 (this gives the picture in the folder)
 
5) Jij(max_diff(ijk,:),max_diff(i2 j2 k2,:), FA(ijk), FA(i2 j2 k2) )
