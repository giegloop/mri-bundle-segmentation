"""
SLT PROJECT - Jij COMPUTATION

Team members: Sjoerd van Bekhoven, Torres Garcia Moises, Antonio Orvieto

Jij is the cost of considering coupled object i and object j. Here we use the
maximum diffusion directions and the fractional anisotropy of each 3d pixel
as a compression of the original data, as motivated from

Diffusion Tensor MR Imaging and Fiber Tractography: Theoretic Underpinnings
by P. Mukherjee, J.I. Berman, S.W. Chung, C.P. Hess R.G. Henry.
"""
from scipy import spatial as sc

def J_ij(veci,vecj,FAi,FAj):
    if (and(FAi<0.34, FAj<0.34):
       return 0 
    else :
       return sc.distance.cosine(veci, vecj)+1
            
    
