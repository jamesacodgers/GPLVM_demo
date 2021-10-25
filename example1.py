# %%
import numpy as np 
import numpy.random as rand
import numpy.linalg as lin 
import matplotlib.pyplot as plt


D = 2                           # Number of observed dimensions 
q = 1                           # Number of latent dimensions
N = 40                         # Number of data points

beta = 1e-1                       # Scale parameter for variance of noise 

X = rand.normal(0,1,N)          # Generate true latent variables

W = np.array([1,1])             # Linear mapping from x to y 

Y = np.outer(X,W) + rand.normal(0,1/np.sqrt(beta), (N,2))               # Map to other latent variable 

print(X,Y[:,0],Y[:,1])


plt.figure()
plt.scatter(Y[:,0],Y[:,1])
plt.show()
# %%

eigval, eigvec = lin.eig(Y @ Y.T)

i = np.argmax(np.real(eigval))



X_est = np.real(eigvec[:,i])*(np.real(eigval[i])/D)**(-1/2)
X_prior = np.real(eigvec[:,i])*(np.real(eigval[i])/D - 1/beta)**(-1/2)
#print(X_est, (np.real(eigval[i]) - 1/beta)**(1/2))

plt.figure()
plt.scatter(X,-X_est)
plt.scatter(X,-X_prior)
plt.xlabel("true")
plt.ylabel("found")
plt.plot(np.linspace(-3,3),np.linspace(-3,3), c = "orange")
plt.show()



# %%

U, S, VT = lin.svd(Y,False)

X_est_svd = U[:,0] * S[0]**-1/2
X_prior_svd = U[:,0] * (S[0]-1/beta)**-1/2


plt.figure()
plt.scatter(X,X_est_svd)
plt.scatter(X,X_prior_svd)
plt.scatter(X,-X_est)
plt.scatter(X,-X_prior)
plt.xlabel("true")
plt.ylabel("found")
#plt.plot(np.linspace(-3,3),np.linspace(-3,3), c = "orange")
plt.show()

# %%

plt.figure()
#plt.scatter(X_est_svd,X_prior_svd)
plt.scatter(X_est,X_prior)
plt.plot(np.linspace(min(X_est),max(X_est)),
        np.linspace(min(X_est),max(X_est)), c = "black")
plt.show()


# %%
