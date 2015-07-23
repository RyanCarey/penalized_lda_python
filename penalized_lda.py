import numpy as np
import pandas as pd
import scipy
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

def within_class_sds(X,y):
    # gets within-class standard deviations for each column
    n,p = X.shape
    classes = sorted(set(y))
    centroids = get_centroids(X,y)
    y_integer = [np.where(np.array(classes)==i)[0][0] for i in y]
    deviations = X - np.array([centroids[i] for i in y_integer])  
    within_class_sds = np.sqrt((deviations**2).mean(axis=0))
    return(within_class_sds)
    # give every feature mean zero and within-class standard deviation of 1

def factor_to_integer(y):
    assert type(y) is np.ndarray, 'y is not a numpy array'
    y_new = np.zeros_like(y,dtype=int)
    classes = sorted(set(y))
    for (i_cl,cl) in enumerate(classes):
        y_new[y==cl]=i_cl
    return y_new

def get_centroids(X,y):
    classes = sorted(set(y))
    centroids = np.array([X[y==cl,].mean(axis=0) for cl in classes]) 
    return(centroids)   

def classify(Xtr,ytr,Xte):
    # classifies each example in Xte to the nearest centroid
    classes = sorted(set(ytr))
    prior = np.array([(ytr==k).mean() for k in classes])
    centroids = get_centroids(Xtr,ytr)
            # classes x parameters; each row is the centroid of a class
    displacements = Xte[:,np.newaxis,:] - centroids[np.newaxis,:,:]
            # examples x classes x parameters
    squared_distances = (displacements**2).sum(axis=2)
            # examples x classes; each row is distance of example from each centroid
    posterior = -.5*squared_distances + np.log(prior)
    predictions = np.argmax(posterior,axis=1)
    predictions = np.array([classes[i] for i in predictions])
    return(predictions)

def utility_pca(X, P, v, reg, d):
    '''computes the utility function for penalized matrix decomposition, 
    where X is approximated as \sum_k=1^K{d_k x u_k x v_k} with L1 regularizion of v_k.'''
    regularization = reg*abs(v).sum()
    utility = v.T.dot(X.T).dot(P).dot(X).dot(v)-d*regularization
    return(utility)

def rectifier(a):
    a = np.copy(a)
    a[a<0] = 0
    return(a)

def soft_threshold(a,c):
    return(np.sign(a)*rectifier(abs(a)-c))

class DivergingError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)
    
def penalized_pca(X,reg,K,max_iter=20):
    n,p = X.shape
    
    betas = np.zeros((p,K))
       
    for k in range(K):
        
        if k>0:
            U_a, d_a, V_a = np.linalg.svd(X.dot(betas))
            u = U_a[:,d_a>(1e-10)]  #left singular vectors with nonzero singular values
            P = np.eye(n) - u.dot(u.T)
        elif k==0:
            P = np.eye(n)
        
        U_x, d_x, V_x = scipy.sparse.linalg.svds(X.T.dot(P),k=1)
        d = d_x[0]**2
        beta = U_x[:,0]
    
        utils = []
        for i in range(max_iter):
            if ((len(utils)<4) or (abs(utils[-1]-utils[-2])/max(1e-3,utils[-1])) > (1e-6)
               ) and (abs(beta).sum()>0):
               tmp = X.T.dot(P).dot(X.dot(beta))
               beta = soft_threshold(tmp,d*reg/2)
               if np.linalg.norm(beta)!=0:
                   beta /= np.linalg.norm(beta)
               utils.append(utility_pca(X,P,beta,reg,d))
        betas[:,k] = beta
    if (len(utils)>2) and ((np.array(utils[1:])-np.array(utils)[:-1]).min() < -1e-6):
        raise NotConvergingError('Utility is decreasing')            
    return(betas)

def one_hot(y):
    if (type(y) is not np.ndarray) or (len(y.shape)!=1):
        raise ValueError('y must be a 1D numpy array')
    y = factor_to_integer(y)
    y_new = np.zeros((len(y),len(set(y))))
    y_new[range(len(y)),y] = 1
    return(y_new)

def get_fold_indices(filename='/home/ryan/ml/dm/folds_new.csv'):
    folds = pd.read_csv(filename,index_col=0)
    folds = np.array(folds_dat).T
    folds = [(i[~np.isnan(i)]-1).astype(int) for i in folds]
    return(folds)

class PenalizedLDA:
    def __init__(self,reg = None, K = None, max_iter = 20):        
        self.reg = reg
        self.K = K
        self.max_iter = max_iter
    def fit(self,Xtr,ytr,standardized=False):
        classes = sorted(set(ytr))
        if np.isnan(Xtr).any():
            raise ValueError('Xtr must not have any missing (NaN) values')
        if ytr.dtype != int:
            raise ValueError('ytr must be an array of integers')
        if np.isnan(ytr).any():
            raise ValueError('ytr must not have any missing (NaN) values')
        if self.K!=None and self.K >= len(set(ytr)):
            raise ValueError('K must be less than the number of unique classes')
        if self.K is None:
            self.K = len(set(ytr))-1

        n,p = Xtr.shape
        if not standardized:
            self.wcsds_ = within_class_sds(Xtr,ytr)
            if 0 in self.wcsds_:
                raise ValueError('Some features have 0 within-class standard deviation')
            else:
                self.means_ = Xtr.mean(axis=0)
                Xtr = (Xtr-self.means_)/self.wcsds_
        sqrt_cov_between = ((np.sqrt(one_hot(ytr)/one_hot(ytr).sum(axis=0))).T.dot(Xtr)/
                             np.sqrt(n))
        self.discrim_ = penalized_pca(sqrt_cov_between, reg=self.reg, K=self.K,max_iter=20)
        self.discrim_[np.isnan(self.discrim_)] = 0
        self.Xtr_proj_ = Xtr.dot(self.discrim_)
        self.ytr_ = ytr
        return(self)

    def predict(self,Xte,standardized=False,k=None):
        if k!=None and k>self.K:
            raise ValueError('number of dimensions used for classification (k) must be '+ 
                     'less than or equal to the dimensionality of the projection (K)')
        if k==None: 
            #by default, use all discriminant vectors for prediction
            k = self.K
        if not standardized:
            #set mean to zero and within-class variance to one
            Xte = (Xte-self.means_)/self.wcsds_

        #project to dimensionality k using discriminant
        Xte_projected = Xte.dot(self.discrim_[:,:k])

        #predict class as nearest centroid in projected space
        y_predicted = classify(self.Xtr_proj_[:,:k],self.ytr_,Xte_projected)
        return(y_predicted)
    def error(self,Xte,yte,k=None,standardized=False):
        if yte.dtype != int:
            raise ValueError('yte must be an array of integers')
        err = (self.predict(Xte, k=k, standardized=standardized)!=yte).sum()
        return(err)
    def __repr__(self):
        return "Penalized LDA(reg=%s, K=%s, max_iter=%s)" % (self.reg, self.K, self.max_iter)

def penalized_lda_cv(X,y,regs=None, K=None, n_folds=6, folds=None):
    if folds is None:
        folds = StratifiedKFold(y,n_folds)
    if y.dtype != int:
        raise ValueError('y must be an array of integers')
    if regs is None:
        regs = np.logspace(.1,10,5)
        
    # if only one K value needs to be used
    if K is not None or len(set(y))==2:
        if len(set(y))==2:
            K = 1
        err = np.zeros((len(folds),len(regs)))
        nonzero_betas = np.zeros((len(folds),len(regs)))
        
        for (i_fold, (tr,val)) in enumerate(folds):
            print('fold',i_fold)
            Xtr = X[tr,:]
            ytr = y[tr]
            Xval = X[val,:]
            yval = y[val]
            
            # scale data using mean and within-class standard deviation from training set
            wcsds = within_class_sds(Xtr,ytr)
            if 0 in wcsds:
                raise ValueError('Some features have 0 within-class standard deviation')
            else:
                means = Xtr.mean(axis=0)
                Xtr = (Xtr - means) / wcsds
                Xval = (Xval - means) / wcsds
            
            for (i_reg, reg) in enumerate(regs):
                print('lambda',reg)
                discrim = penalized_lda_fit(Xtr,ytr,reg,K=K,standardized=True)[1]
                y_pred = lda_predict(Xtr,ytr,Xval,discrim,K)
                print('predicted y:')
                print(y_pred)
                err[i_fold,i_reg] = (y_pred!=yval).sum()
                nonzero_betas[i_fold,i_reg] = (discrim!=0).any(axis=1).sum(axis=0)
        err_mean = err.mean(axis=0)
        nonzero_mean = nonzero_betas.mean(axis=0)
        reg_best = regs[err_mean.argmin()]
        return(err_mean, nonzero_mean, reg_best, K,nonzero_betas,err)
    
    # if it is necessary to cross-validate over K values
    else:
        Ks = [i for i in range(1,len(set(y)))]
        err = np.zeros((len(folds),len(regs),len(Ks)))
        nonzero_betas = np.zeros((len(folds),len(regs),len(Ks)))
        for (i_fold, (tr,val)) in enumerate(folds):
            print('fold',i_fold)
            Xtr = X[tr,:]
            ytr = y[tr]
            Xval = X[val,:]
            yval = y[val]
            
            # scale data using mean and within-class standard deviation from training set
            wcsds = within_class_sds(Xtr,ytr)
            if 0 in wcsds:
                raise ValueError('Some features have 0 within-class standard deviation')
            else:
                means = Xtr.mean(axis=0)
                Xtr = (Xtr - means) / wcsds
                Xval = (Xval - means) / wcsds
            
            for (i_reg, reg) in enumerate(regs):
                print('lambda',reg)
                discrim = penalized_lda_fit(Xtr,ytr,reg,K=max(Ks),
                                                standardized=True)[1]
                for i_k,k in enumerate(Ks):
                    ypred = lda_predict(Xtr,ytr,Xval,discrim,k)
                    print('predicted y:')
                    print(ypred)
                    err[i_fold,i_reg,i_k] = (ypred!=yval).sum()
                    nonzero_betas[i_fold,i_reg,i_k] = discrim[:,:k].any(axis=1).sum(axis=0)
    
        err_mean = err.mean(axis=0)
        nonzero_mean = nonzero_betas.mean(axis=0)
        i_best_params = np.unravel_index(err_mean.argmin(),err_mean.shape)
        reg_best = regs[i_best_params[0]]
        K_best = Ks[i_best_params[1]]
        return(err_mean, nonzero_mean, reg_best, K_best,nonzero_betas,err)
