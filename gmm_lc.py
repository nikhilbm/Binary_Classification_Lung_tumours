import numpy as np
import math
from numpy import ndarray
from numpy import random as r

#--------------------------------------------------------------------------------------------------
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K,n):
    # Initialize to K random centers
    idx=r.randint(nf,size=K)
    oldmu=f[idx,:]
    idx=r.randint(nf,size=K)
    mu=f[idx,:]
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return mu
#============================================================================================
av=np.zeros(25)
at=np.zeros(25)
ev=np.zeros(25)
et=np.zeros(25)
for u in range(25):
    #class 1 training
    #------------------------
    n=(u+1)*10 #no of feature vectors
    ng=u+1 #no of gausians
    nf=9 #feature dimension
    n1=n            
    #k-means algorithem to form clusters
    f=np.genfromtxt('train0.csv',delimiter=',')
    f=f[0:n1,:]
    idx=r.randint(n,size=ng)
    #g=f[idx,:]
    g=f[0:ng,:]
    z=find_centers(f, ng,n)
    g=np.array(z)
    ng=len(z)
    ng1=ng

    ngs=np.zeros((ng,nf))
    ngc=np.zeros(ng)
    newg=np.zeros((ng,nf))
    c=np.zeros(n)

    #g=r.rand(ng,nf)
    #g=f[0:5,:]

    t= 1
    while t:
        ngs=np.zeros((ng,nf))
        ngc=np.zeros(ng)
        newg=np.zeros((ng,nf))
        c=np.zeros(n)
        cl=0
        for i in range(n):
            m=sum(pow((f[i,:]-g[0,:]),2))
            for j in range(ng):
                mint=sum(pow((f[i,:]-g[j,:]),2))
                if mint<m:
                    m=mint
                    cl=j
            c[i]=cl
            ngs[cl,:]=ngs[cl,:]+f[i,:]
            ngc[cl]=ngc[cl]+1
        #print c
        for j in range(ng):
            if ngc[j] != 0:
                newg[j,:]=ngs[j,:]/ngc[j]
        t=np.any(abs(g-newg)>np.zeros((ng,nf)))
        g=newg #means
    g1=g
    w1=ngc/n #weights
    vs=np.zeros((ng,nf))
    cov1=np.zeros((ng,nf,nf))
    for i in range(n):
            x=int(c[i])
            vs[x,:]=vs[x,:]+pow((g[x,:]-f[i,:]),2)
    for j in range(ng):
            if ngc[j] != 0:
                vs[j,:]=vs[j,:]/ngc[j]+0.00001*np.ones((1,nf))
            cov1[j,:,:] = np.diag(vs[j,:]) #covariance matrices
    #-------------------------------------------------------------------------------
    #class 2 training
    #-------------------------------------------------------------------------------
    n=(u+1)*10 #no of feature vectors
    ng=u+1 #no of gausians
    nf=9 #feature dimension
    n2=n
    #k-means algorithem to form clusters
    f=np.genfromtxt('train1.csv',delimiter=',')
    f=f[0:n2,:]
    idx=r.randint(n,size=ng)
    #g=f[idx,:]
    g=f[0:ng,:]
    z=find_centers(f, ng,n)
    g=np.array(z)
    ng=len(z)
    ng2=ng

    ngs=np.zeros((ng,nf))
    ngc=np.zeros(ng)
    newg=np.zeros((ng,nf))
    c=np.zeros(n)
       
    #g=r.rand(ng,nf)
    #g=f[0:5,:]

    t= 1
    while t:
        ngs=np.zeros((ng,nf))
        ngc=np.zeros(ng)
        newg=np.zeros((ng,nf))
        c=np.zeros(n)
        cl=0
        for i in range(n):
            m=sum(pow((f[i,:]-g[0,:]),2))
            for j in range(ng):
                mint=sum(pow((f[i,:]-g[j,:]),2))
                if mint<m:
                    m=mint
                    cl=j
            c[i]=cl
            ngs[cl,:]=ngs[cl,:]+f[i,:]
            ngc[cl]=ngc[cl]+1
        for j in range(ng):
            if ngc[j] != 0:
                newg[j,:]=ngs[j,:]/ngc[j]
        t=np.any(abs(g-newg)>np.zeros((ng,nf)))
        g=newg #means
    g2=g
    w2=ngc/n #weights
    vs=np.zeros((ng,nf))
    cov2=np.zeros((ng,nf,nf))
    for i in range(n):
            x=int(c[i])
            vs[x,:]=vs[x,:]+pow((g[x,:]-f[i,:]),2)
    for j in range(ng):
            if ngc[j] != 0:
                vs[j,:]=vs[j,:]/ngc[j]+0.00001*np.ones((1,nf))
            cov2[j,:,:] = np.diag(vs[j,:]) #covariance matrices

    #----------------------------------------------------------------
    #validation class 1 for validation
    #----------------------------------------------------------------
    n=150#(u+1)*5 #no of feature vectors
    nf=9 #feature dimension
    f=np.genfromtxt('v0.csv',delimiter=',')
    f=f[0:n,:]
    p1=0
    p2=0
    c1count=0
    c2count=0
    for i in range(n):
        p1=0
        p2=0
        for j in range(ng1):
            if w1[j]!=0:
                inv = np.linalg.inv(cov1[j])
                det = np.linalg.det(cov1[j])
                t4 = np.matrix(f[i,:]-g1[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p1 = p1 + w1[j] * 1.0/t7 * math.exp(-t8/2)

        for j in range(ng2):
            if w2[j]!=0:
                inv = np.linalg.inv(cov2[j])
                det = np.linalg.det(cov2[j])
                t4 = np.matrix(f[i,:]-g2[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p2 = p2 + w2[j] * 1.0/t7 * math.exp(-t8/2)

        if n1*p1>=n2*p2:
            c1count = c1count + 1
        else:
            c2count = c2count + 1
    tn = c1count
    fp = c2count
    #----------------------------------------------------------------
    #validation class 2 for validation
    #----------------------------------------------------------------
    n=50#(u+1)*2 #no of feature vectors
    nf=9 #feature dimension
    f=np.genfromtxt('v1.csv',delimiter=',')
    f=f[0:n,:]
    p1=0
    p2=0
    c1count=0
    c2count=0
    for i in range(n):
        p1=0
        p2=0
        for j in range(ng1):
            if w1[j]!=0:
                inv = np.linalg.inv(cov1[j])
                det = np.linalg.det(cov1[j])
                t4 = np.matrix(f[i,:]-g1[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p1 = p1 + w1[j] * 1.0/t7 * math.exp(-t8/2)

        for j in range(ng2):
            if w2[j]!=0:
                inv = np.linalg.inv(cov2[j])
                det = np.linalg.det(cov2[j])
                t4 = np.matrix(f[i,:]-g2[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p2 = p2 + w2[j] * 1.0/t7 * math.exp(-t8/2)

        if p1>=p2:
            c1count = c1count + 1
        else:
            c2count = c2count + 1
    fn = c1count
    tp = c2count
    accuracy = (tp+tn)*100/(tp+tn+fp+fn)
    av[u]=accuracy
    ev[u]=100-accuracy
#----------------------------------------------------------------
    #validation class 1 for training
    #----------------------------------------------------------------
    n=(u+1)*10 #no of feature vectors
    nf=9 #feature dimension
    f=np.genfromtxt('train0.csv',delimiter=',')
    f=f[0:n,:]
    p1=0
    p2=0
    c1count=0
    c2count=0
    for i in range(n):
        p1=0
        p2=0
        for j in range(ng1):
            if w1[j]!=0:
                inv = np.linalg.inv(cov1[j])
                det = np.linalg.det(cov1[j])
                t4 = np.matrix(f[i,:]-g1[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p1 = p1 + w1[j] * 1.0/t7 * math.exp(-t8/2)

        for j in range(ng2):
            if w2[j]!=0:
                inv = np.linalg.inv(cov2[j])
                det = np.linalg.det(cov2[j])
                t4 = np.matrix(f[i,:]-g2[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p2 = p2 + w2[j] * 1.0/t7 * math.exp(-t8/2)

        if n1*p1>=n2*p2:
            c1count = c1count + 1
        else:
            c2count = c2count + 1
    tn = c1count
    fp = c2count
    #----------------------------------------------------------------
    #validation class 2 for training
    #----------------------------------------------------------------
    n=(u+1)*10 #no of feature vectors
    nf=9 #feature dimension
    f=np.genfromtxt('train1.csv',delimiter=',')
    f=f[0:n,:]
    p1=0
    p2=0
    c1count=0
    c2count=0
    for i in range(n):
        p1=0
        p2=0
        for j in range(ng1):
            if w1[j]!=0:
                inv = np.linalg.inv(cov1[j])
                det = np.linalg.det(cov1[j])
                t4 = np.matrix(f[i,:]-g1[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p1 = p1 + w1[j] * 1.0/t7 * math.exp(-t8/2)

        for j in range(ng2):
            if w2[j]!=0:
                inv = np.linalg.inv(cov2[j])
                det = np.linalg.det(cov2[j])
                t4 = np.matrix(f[i,:]-g2[j,:])
                t5 = np.matrix(inv)
                t7 = math.sqrt(pow(2*math.pi, nf) * det)
                t8 = np.dot(np.dot(t4, t5), t4.T)
                p2 = p2 + w2[j] * 1.0/t7 * math.exp(-t8/2)

        if p1>=p2:
            c1count = c1count + 1
        else:
            c2count = c2count + 1
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
plt.title('learning curve for training data-gmm')
plt.show()

plt.plot(ev)
plt.ylabel('error')
plt.xlabel('traning data')
plt.title('learning curve for validation data-gmm')
plt.show()

