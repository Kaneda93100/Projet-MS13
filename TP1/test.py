import numpy as np

M = np.array([[1,-1],
              [-1,1]])

VAP, VEP = np.linalg.eigh(M)

print(f"Vecteurs propres : {VEP}\n\n")
print(f"Valeurs propres : {VAP}\n\n")

print(f"{M@VEP[:,0] - VAP[0]*VEP[:,0]}\n\n{M@VEP[:,1] - VAP[1]*VEP[:,1]}")
print(f"{M@VEP[0,:] - VAP[0]*VEP[0,:]}\n\n{M@VEP[1,:] - VAP[1]*VEP[1,:]}")
