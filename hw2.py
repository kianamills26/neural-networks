#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:53:38 2019

@author: kianamills
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

# PROBLEM ONE
def outer_product(f, g, n):
    # initialize a matrix of given dimensions
    A = np.zeros((n,n))
    #print(A)
    for i in range(n):
        for j in range(n):
            A[i][j] = g[i] * f[j]
    return A
    
# Generates n random vector pairs with dimensionality dim
def generate_vector_pairs(n, dim):
    F = []
    G = []
    for _ in range(n):
        f = np.random.rand(dim,1)
        g = np.random.rand(dim,1)
        # centered
        f = f - f.mean()
        g = g - g.mean()
        # scaled
        f = f / np.linalg.norm(f)
        g = g / np.linalg.norm(g)
        if n == 1:
            return f,g
        else:
            F.append(f)
            G.append(g)
    return F,G

# Destroys proportion p of matrix A at random
def ablate(p, A):
    if p <= 0:
        return A
    if p == 1:
        return np.zeros((A.shape[0],A.shape[0]))
    n = int(p * A.shape[0])
    print(n)
    A_new = A.copy()
    for i in range(n):
        x = np.random.randint(low=0, high=A.shape[0])
        y = np.random.randint(low=0, high=A.shape[0])
        A_new[x][y] = 0
        
    return A_new
    

# PROBLEM ONE
dim = 100

f, g = generate_vector_pairs(1, dim)
A = outer_product(f, g, dim)
#print(A)

# Show that Af giives an output g' that is the same direction as g
g_prime = A.dot(f)

# Show that dot product is 1 (angle between is zero degrees) and that the length of g' is 1
dot_product = np.sum(g_prime * g)
length = np.linalg.norm(g_prime)

print("Dot Product: ", dot_product)  # Dot product is 1, so cosine of the angle = 1
print("Length of g': ", length)   # Length of g' is approx one
 

# PROBLEM TWO
# Generate a new normalized vector, f'
f_prime = np.random.rand(dim,1)
f_prime = f_prime - f_prime.mean()
f_prime = f_prime / np.linalg.norm(f_prime)

# Check to see they are more or less orthogonal
dot_product = np.sum(f * f_prime )
print("Dot Product of f and f' : ", dot_product)

# Angle between is approx 90 degrees
angle_rad = np.arccos(dot_product)
angle_deg = angle_rad * (180/np.pi)
print("Angle between f and f' : ", angle_deg)

# Compute Af' and its length
Af = A.dot(f_prime)
length_Af = np.linalg.norm(Af)
print("Length of Af': ", length_Af) # The length of Af' is approx zero



# PROBLEM THREE
N = [1,20,40,60,80,100]
mean_dot_products = []
std_dot_products = []
mean_g_lengths = []

for x in range(6):
    
    dim = 100
    n = N[x] 
    print("N = ", n)
    F, G = generate_vector_pairs(n, dim)

    if n == 1:
        A = outer_product(F, G, dim)  
        # Compare output for the first (and only) vector f
        g_prime = A.dot(F)
        dot_product = np.sum(g_prime * G)
        mean_dot_products.append(dot_product)
        print("Dot Product of g1 and g'1: ",  dot_product)
        print("Length of g'1: ", np.linalg.norm(g_prime))
    else:
        A_i = [ outer_product(F[i], G[i], dim) for i in range(n) ]
        # Form the overall connectivity matrix A
        A = np.zeros((dim,dim))
        for i in range(n):
            A = np.add(A, A_i[i])
        
        # Compare output for the first vector f1
        g_prime = A.dot(F[0])
        dot_product = np.sum(g_prime * G[0])
        print("Dot Product of g1 and g'1: ",  dot_product)
        print("Length of g'1: ", np.linalg.norm(g_prime))

        # Compute the output for each stored vector fi
        g_prime = [ A.dot(F[i]) for i in range(n) ]
        g_lengths = [ np.linalg.norm(g_prime[i])  for i in range(n) ]
        dot_products = [ np.sum(g_prime[i] * G[i]) for i in range(n) ]
        mean_dot_products.append(np.mean(dot_products))
        std_dot_products.append(np.std(dot_products))
        mean_g_lengths.append(np.mean(g_lengths))
        print("Mean and St. Dev Dot Products: ", np.mean(dot_products), np.std(dot_products))
        #print("St. Dev of Dot Products: ", np.std(dot_products))
        print("Mean of lengths of g': ", np.mean(g_lengths))

    # Generate a new set of n random vectors
    H = []
    for _ in range(n):
        h = np.random.rand(dim,1)
        h = h - h.mean()
        h = h / np.linalg.norm(h)
        H.append(h)

    h_prime = [ A.dot(H[i]) for i in range(n) ]
    h_lengths = [ np.linalg.norm(h_prime[i]) for i in range(n) ]
    print("Mean of lengths of h': ", np.mean(h_lengths) )  
    
    # d) (iv) Plotting g lengths and h lengths
    plt.figure()
    bins = np.linspace(0, 2, 30)
    plt.hist([g_lengths, h_lengths], bins, alpha=0.7, label=['g lengths','h\' lengths'])
    plt.legend(loc='upper right')
    plt.title("Histogram of Lengths of h' and Lengths of g, N= %d" % (n))
    plt.show()
    


# bar graph of mean dot product of g and g'
objects = (1,20,40,60,80,100)
y_pos = np.arange(len(objects))

plt.bar(y_pos, mean_dot_products, align='center', alpha=0.7)
plt.xticks(y_pos, objects)
plt.ylim(0.8, 1.1)
plt.ylabel('Dot Product Value')
plt.xlabel('Number of Vector Pairs')
plt.title('Mean Dot Products')
plt.show()
    



# PROBLEM 4 
# part a: Destroy parts of the matrix at random, by setting random indexes to zero
# Destroy 10%, 20%, 30%, 40%, 50%, Use n = 50 vectors
n = 50
dim = 100
p = 1
F, G = generate_vector_pairs(n, dim)

A_i = [ outer_product(F[i], G[i], dim) for i in range(n) ]
# Form the overall connectivity matrix A
A = np.zeros((dim,dim))
for i in range(n):
    A = np.add(A, A_i[i])
    
# PRE-ABLATED: use f or newly generated h?
# Compute the output for each stored vector fi
g_prime = [ A.dot(F[i]) for i in range(n) ]
#g_prime = g_prime / np.linalg.norm(g_prime) ## if leave out this line, dotproduct = 1, else dotproduct ~0
dot_products = [ np.sum(g_prime[i] * G[i]) for i in range(n) ]
print("Mean of Dot Products: ", np.mean(dot_products))
print("St. Dev of Dot Products: ", np.std(dot_products))

A_new = ablate(p, A)
np.array_equal(A, A_new)

# POST-ABLATED
g_prime_ab = [ A_new.dot(F[i]) for i in range(n) ]
#g_prime_ab = g_prime_ab / np.linalg.norm(g_prime_ab)
dot_products = [ np.sum(g_prime_ab[i] * G[i]) for i in range(n) ]
print("Mean of Dot Products: ", np.mean(dot_products))
print("St. Dev of Dot Products: ", np.std(dot_products))

# Measure the damage between pre- and post-ablated outputs
damage = [ np.sum(g_prime[i] * g_prime_ab[i]) for i in range(n) ]
print("Mean damage (cosine) btwn pre- and post-ablated outputs: ", np.mean(damage))
print("Mean angle btwn pre- and post-ablated outputs: ", np.mean(np.arccos(damage)* (180/np.pi)))










