from numpy import exp

# return 1./(1+np.exp(-A))

def activation_fn(A):
    expA = exp(-A)
    return 1./(1+expA)