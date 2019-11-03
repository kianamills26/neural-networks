# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem One
# This function generates n random numbers, uniformly distributed
def gen_uniform(n):
    np.random.seed(2)
    
    x = np.random.rand(n)

    plt.figure()
    _ = plt.hist(x, bins=10)
    plt.title('Uniform Distribution')
    plt.show()
    
# Problem Two
# This function generates n random numbers, normally distributed
def gen_normal(n):
    np.random.seed(2)
    
    mu, sigma = 0, 0.1
    x = np.random.normal(mu, sigma, n)
    
    plt.figure()
    _ = plt.hist(x, bins=40, density=True)
    plt.title('Normal Distribution')
    plt.show()
    
# Problem Three
# This function generates many pairs of vectors and their dot products
def gen_dotproducts(n):
    
    X = np.random.rand(n,1)
    Y = np.random.rand(n,1)
    
    # centered
    X_centered = X - X.mean()
    Y_centered = Y - Y.mean()
    
    # scaled
    X_normed = X_centered / np.linalg.norm(X_centered)
    Y_normed = Y_centered / np.linalg.norm(Y_centered)
    
    X_length = np.linalg.norm(X_normed)
    Y_length = np.linalg.norm(Y_normed)
         
    # Dot Product of 2 normalized vectors is equal to the cosine angle
    dot_product = np.sum(X_normed * Y_normed)

    # Angle between the vectors = 90 deg
    angle_rad = np.arccos(dot_product)
    angle_deg = angle_rad * (180/np.pi)
    #print(angle_deg)
    
    return dot_product
    
# Problem Four
def estimate_pi(n):
    within_circle = 0
    total = n
    
    for _ in range(n):
        # Compute 2 random numbers (x,y) uniformaly distributed between -1 and 1
        x,y = np.random.uniform(-1.0, 1.0, 2)
        #print(x,y)
        
        # Decide if (x,y) is inside or outside the circle
        dist = np.sqrt(x**2 + y**2)
        if dist <= 1:
            within_circle += 1
    
    # Ratio of total to within is equal to ratio of the areas, ie 4/pi
    pi = (4 * within_circle)/ total
    return pi
    
# PROBLEM 1
n = 50000
gen_uniform(n)


# PROBLEM 2
gen_normal(n)


# PROBLEM 3
dim = 1000
iterations = 10000
dot_products = [ gen_dotproducts(dim) for i in range(iterations) ]

#print(dot_products)
print("mean: ", np.mean(dot_products))
print("std: ", np.std(dot_products))
print("var: ", np.var(dot_products))

plt.figure()
_ = plt.hist(dot_products, bins = 'auto')
plt.title('Histogram of Dot Products, Dimension = %d' % (dim))
plt.show()
    
# PROBLEM 4
p = []
n = 100
for _ in range(5):
    p.append(estimate_pi(n))
    n = n * 10
    
print(p)

