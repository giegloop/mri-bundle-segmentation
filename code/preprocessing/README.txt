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
   
4) FA(i*j*k) (anisotropy) gives you an important information : is the 3d point white matter or not.
   Not all data points need to be considered in the clustering, only white matter. to do this you need to set
   a threshold. To get a nice result pick only data point with F>0.34 (this gives the picture in the folder)
 
4) The cost Jij is a J_proximity * J_shape (product). The function J_shape only returns J_shape.
	Jij will be a number from 1 (perfectly similar) to 2 (opposite). The bias can be different (if we set J=0
	if they are similar, equal shapes will be clustered together for sure even if they are far away, as we pay 0)
	
	If you need Jshape btw two points 1 and 2 just call JShape(max_diff(i1*j1*k1,:),max_diff(i1*j1*k1,:))
	
	There is still the need to set up a proper J_proximity. a good answer is in
	http://vyssotski.ch/BasicsOfInstrumentation/SpikeSorting/Blatt1996(SuperParamagneticClusteringOfData).pdf
	formula(2) (I cannot do it because I do not know the neighborhood <i,j> criterion you have set.)