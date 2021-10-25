# %%
import numpy as np 
import numpy.random as rand
import numpy.linalg as lin 
import matplotlib.pyplot as plt


D = 2                           # Number of observed dimensions 
q = 1                           # Number of latent dimensions
N = 100                         # Number of data points

beta = 0.1                       # Scale parameter for variance of noise 

X = rand.normal(0,1,N)          # Generate true latent variables

W = np.array([np.sqrt(8),np.sqrt(8)])             # Linear mapping from x to y 

Y = np.outer(X,W) + rand.normal(0,1/beta, (N,2))               # Map to other latent variable 

print(X,Y[:,0],Y[:,1])


plt.figure()
plt.scatter(Y[:,0],Y[:,1])
plt.show()
# %%

eigval, eigvec = lin.eig(Y @ Y.T)

i = np.argmax(np.real(eigval))



X_est = np.real(eigvec[:,i])*(np.real(eigval[i])/D)**(1/2)
X_prior = np.real(eigvec[:,i])*(np.real(eigval[i])/D - 1/beta)**(1/2)
#print(X_est, (np.real(eigval[i]) - 1/beta)**(1/2))

plt.figure()
plt.scatter(X,X_est)
plt.scatter(X,X_prior)
plt.xlabel("true")
plt.ylabel("found")
plt.plot(np.linspace(-3,3),np.linspace(-3,3), c = "orange")
plt.show()



# %%
