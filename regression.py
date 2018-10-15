import numpy as np
import math
from numpy import ndarray
from numpy import random as r

n=100 #no of feature vectors
nf=9 #feature dimension
f=np.genfromtxt('t.csv',delimiter=',')

x=np.ones((n,(nf+1)))
x[:,1:(nf+1)]=f[240:340,0:nf]/np.max(f)
y=np.ones((n,1))
y[:,0]=f[240:340,nf]

t=np.zeros(((nf+1),1))
l=0.09
a=.01
#a=1/n
deriv=np.zeros(nf+1)
cond=1
#for j in range(1):
while cond:
    for i in range(nf+1):
        u=np.ones((n,1))
        u[:,0]=x[:,i]
        temp1=-np.dot(x,t)
        h=np.zeros((n,1))
        for j in range(n):
            h[j,:]=1/(1+math.exp(temp1[j,:]))
        #h=np.ones((n,1))/(np.ones((n,1))+math.exp(-np.dot(x,t)))
        deriv[i] = sum((h-y)*u)+l*t[i]
        t[i]=t[i]-a*deriv[i]
    cond = np.any(abs(deriv)>.0001*np.ones(nf+1))




#----------------------------------------------------------------
#validation class 1
#----------------------------------------------------------------
n=56 #no of features
nf=9 #feature dimension
f=np.genfromtxt('testing0.csv',delimiter=',') 
y=0
c1count=0
c2count=0
for i in range(n):
    y=np.dot(np.concatenate((np.ones(1),f[i,:]),axis=0),t)
    if y>=0:
        c2count = c2count + 1
    else:
        c1count = c1count + 1
tn = c1count
fp = c2count
#----------------------------------------------------------------
#validation class 2
#----------------------------------------------------------------
n=20 #no of features
nf=9 #feature dimension
f=np.genfromtxt('testing1.csv',delimiter=',')
y=0
c1count=0
c2count=0
for i in range(n):
    y=np.dot(np.concatenate((np.ones(1),f[i,:]),axis=0),t)
    if y>=0:
        c2count = c2count + 1
    else:
        c1count = c1count + 1
fn = c1count
tp = c2count    
#-------------------------------------------------------------------
#results
#-------------------------------------------------------------------
accuracy = (tp+tn)*100/(tp+tn+fp+fn)
sensitivity = tp*100/(tp+fn)
specificity = tn*100/(fp+tn)

print ('true negetives = %d'%tn)
print ('false positives = %d'%fp)
print ('true positives = %d'%tp)
print ('false negetives = %d'%fn)
print ('       ')
print ('accuracy = %d'%accuracy)
print ('sensitivity = %d'%sensitivity)
print ('specificity = %d'%specificity)



