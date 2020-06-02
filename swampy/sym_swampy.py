# Exploring sympy as a way of building up the model
from sympy import *

A_c=Symbol("A_c")
eta_c_np1=Symbol(r"\eta_c^{n+1}")

## 
# Kleptsova, eq 5:
A_c eta[c,n+1]

# a_j^n is 
a[j] = sum(c,  delta[j,c] alpha[j,c] 
