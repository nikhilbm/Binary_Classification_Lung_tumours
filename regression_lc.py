import numpy as np
import math
from numpy import ndarray
from numpy import random as r

av=np.zeros(10)
at=np.zeros(10)
ev=np.zeros(10)
et=np.zeros(10)
for u in range(10):
    n=(u+1)*20 #no of feature vectors
    nf=9 #feature dimension
    f=np.genfromtxt('t.csv',delimiter=',')
    f=f[290-n/2:290+n/2,:]
    x=np.ones((n,(nf+1)))
    for i in range(nf):
        x[:,i]=x[:,i]/max(x[:,i])
    x[:,1:(nf+1)]=f[:,0:nf]/np.max(f)
    y=np.ones((n,1))
    y[:,0]=f[:,nf]

    t=np.zeros(((nf+1),1))
    l=0
    a=.01
    #a=1/n
    deriv=np.zeros(nf+1)
    cond=1
    #for j in range(1):
    while cond:
        for i in range(nf+1):
            p=np.ones((n,1))
            p[:,0]=x[:,i]
            temp1=-np.dot(x,t)
            h=np.zeros((n,1))
            for j in range(n):
                h[j,:]=1/(1+math.exp(temp1[j,:]))
            #h=np.ones((n,1))/(np.ones((n,1))+math.exp(-np.dot(x,t)))
            deriv[i] = sum((h-y)*p)+l*t[i]
            t[i]=t[i]-a*deriv[i]
        cond = np.any(abs(deriv)>.0001*np.ones(nf+1))

    #----------------------------------------------------------------
    #validation class 1 validation
    #----------------------------------------------------------------
    n=150 #no of features
    nf=9 #feature dimension
    f=np.genfromtxt('v0.csv',delimiter=',')
    f=f[0:n,:]
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
    #validation class 2 validation
    #----------------------------------------------------------------
    n=50 #no of features
    nf=9 #feature dimension
    f=np.genfromtxt('v1.csv',delimiter=',')
    f=f[0:n,:]
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
    accuracy = (tp+tn)*100/(tp+tn+fp+fn)
    av[u]=accuracy
    ev[u]=100-accuracy
    #----------------------------------------------------------------
    #validation class 1 training
    #----------------------------------------------------------------
    n=(u+1)*10 #no of features
    nf=9 #feature dimension
    f=np.genfromtxt('train0.csv',delimiter=',')
    f=f[(290-n):290,:]
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
    #validation class 2 training
    #----------------------------------------------------------------
    n=(u+1)*10 #no of features
    nf=9 #feature dimension
    f=np.genfromtxt('train1.csv',delimiter=',')
    f=f[0:n,:]
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
    accuracy = (tp+tn)*100/(tp+tn+fp+fn)
    at[u]=accuracy
    et[u]=100-accuracy
#-------------------------------------------------------------------
#results
#-------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(et)
plt.ylabel('error')
plt.xlabel('traning data')
plt.title('learning curve for training data-reg')
plt.show()

plt.plot(ev)
plt.ylabel('error')
plt.xlabel('traning data')
plt.title('learning curve for validation data-reg')
plt.show()



