import numpy as np
import pandas as pd
import scipy
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

class DivergingError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)
    
def factor_to_integer(y, verbose=False):
    # converts an array of labels into integers
    assert isinstance(y, np.ndarray), 'y must be a numpy array'
    y_new = np.zeros_like(y,dtype=int)
    classes = sorted(set(y))
    for (i_cl,cl) in enumerate(classes):
        y_new[y==cl]=i_cl
    if verbose:
        print('Classes are denoted as follows:')
        print([(i,j) for (i,j) in enumerate(classes)])
    return y_new

def get_centroids(X,y):
    # creates an (n_classes x n_parameters) matrix of centroids
    classes = sorted(set(y))
    centroids = np.array([X[y==cl,].mean(axis=0) for cl in classes]) 
    return(centroids)   

def within_class_sds(X,y):
    # computes within-class standard deviation for each column
    n,p = X.shape
    classes = sorted(set(y))
    centroids = get_centroids(X,y)
    y_integer = [np.where(np.array(classes)==i)[0][0] for i in y]
    deviations = X - np.array([centroids[i] for i in y_integer])  
    within_class_sds = np.sqrt((deviations**2).mean(axis=0))
    return(within_class_sds)
    # give every feature mean zero and within-class standard deviation of 1

def classify(Xtr,ytr,Xte):
    # classifies each example in Xte to the likeliest centroid, taking into
    # account squared distance and a prior based on test-set frequency.
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

def penalized_pca(X,reg,K,max_iter=20):
    ''' Penalized Principal Component Analysis
    Penalized Principal Component Analysis (Penalized PCA) computes a rank-K approximation
    for a matrix X. The matrix is approximated by \sum_k=1^K d_k u_k v_k^T, with an L1
    penalty applied to v_k but not to v_u. The optimization is performed using a minorization
    approach.

    Arguments
    ----------

    X: array
        The matrix to be decomposed

    reg: float 
        The regularization parameter, lambda. A higher value will produce a more sparse solution.

    K: int
        The rank of the approximation for X

    max_iter, default=20:
        The number of iterations taken to optimize this approximation.


    Returns
    ---------
    betas: the v value from the matrix decomposition is returned.

    
    Notes
    ---------
    The penalized matrix decomposition and its application to sparse PCA are 
    described in Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A 
    penalized matrix decomposition, with applications to sparse principal 
    components and canonical correlation analysis." Biostatistics (2009): kxp008.

    at http://biostatistics.oxfordjournals.org/content/early/2009/04/17/biostatistics.kxp008.full

    '''
    n,p = X.shape
    if not (0 < K < p):
        raise ValueError('K must be an integer in the range 0<k<n_parameters')
    
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
    # converts an integer array into an n_examples x k boolean matrix where
    # k is the number of classes, and each row has one-hot encoding.
    if (not isinstance(y, np.ndarray)) or (len(y.shape)!=1):
        raise ValueError('y must be a 1D numpy array')
    y = factor_to_integer(y)
    y_new = np.zeros((len(y),len(set(y))))
    y_new[range(len(y)),y] = 1
    return(y_new)


class PenalizedLDA:
    ''' Penalized Linear Discriminant Analysis.
    Penalized Linear Discriminant Analysis (LDA) is a classifier that finds 
    sparse discrminant vectors, which are used in the setting where the number 
    of parameters is far more than the number of training examples.

    The method is described in Witten, Daniela M., and Robert Tibshirani. 
    "Penalized classification using Fisher's linear discriminant." Journal of 
    the Royal Statistical Society: Series B (Statistical Methodology) 73.5 
    (2011): 753-772. Further information is available at:
    http://faculty.washington.edu/dwitten/Papers/JRSSBPenLDA.pdf
    https://github.com/cran/penalizedLDA/blob/master/R/PenalizedLDA.R


    Parameters
    ----------

    reg: float, default = None
        The regularization parameter, lambda. Higher values give a more sparse 
        solution.

    K: int, default = None
        The number of discriminant vectors used for classification. K must be 
        less than the number of label classes.

    max_iter: int, default = 20
        The number of iterations performed in the minorization procedure used
        to compute the discriminant vectors.


    Attributes
    ----------

    self.discrim_: array, shape = (n_features, K, where K is number of 
    discriminant vectors)
        The discriminant vectors used to project the data.

    self.Xtr_proj_: array, shape = (n_examples, K, where K is the number of 
    discriminant vectors)
        The projection of the training dataset.

    self.ytr_: array, shape = (n_examples,)
        The training labels (which are split further for cross-validation)


    Notes
    ----------
    The algorithm used relies on the fact that Penalized latent discriminant analysis is 
    equivalent to sparse principal component analysis on the between-class covariance 
    matrix, given that the data have already been normalized to have within-class
    covariance of one.


    '''
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
        # predict y values for the given Xte
        if k!=None and k>self.K:
            raise ValueError('number of dimensions used for classification (k) must be '+ 
                     'less than or equal to the dimensionality of the projection (K)')
        if k==None: 
            #by default, use all discriminant vectors for prediction
            k = self.K
        if not standardized:
            #set mean to zero and within-class variance to one
            Xte = (Xte-self.means_)/self.wcsds_

        # project test examples to dimensionality k using discriminant
        Xte_projected = Xte.dot(self.discrim_[:,:k])

        # predict class as likeliest centroid in projected space
        y_predicted = classify(self.Xtr_proj_[:,:k],self.ytr_,Xte_projected)
        return(y_predicted)
    def error(self,Xte,yte,k=None,standardized=False):
        # the count of errors in the prediction made using Xte
        if yte.dtype != int:
            raise ValueError('yte must be an array of integers')
        if len(yte)!= len(Xte):
            raise ValueError('Xte and yte must have same number of examples')
        err = (self.predict(Xte, k=k, standardized=standardized)!=yte).sum()
        return(err)
    def __repr__(self):
        return "Penalized LDA(reg=%s, K=%s, max_iter=%s)" % (self.reg, self.K, self.max_iter)

class PenalizedLDACV:
    ''' Penalized Linear Discriminant Analysis with Cross-validation
    PenalizedLDACV randomly splits the given data into training and cross-
    validation sets and uses these to create a cross-validated PenalizedLDA 
    classifier. Using these parameters, the classifier is trained on all of the
    given data. Afterwards, it is used to predict test data.

    As it is a form of Penalized Linear Discriminant Analysis (LDA), it 
    selects sparse discriminant vectors, and is useful in the setting where
    number of parameters is far more than the number of training examples.

    Penalized LDA is described further in Witten, Daniela M., and Robert 
    Tibshirani. "Penalized classification using Fisher's linear discriminant." 
    Journal of the Royal Statistical Society: Series B (Statistical Methodology) 
    73.5 (2011): 753-772. Further information is available at:
    http://faculty.washington.edu/dwitten/Papers/JRSSBPenLDA.pdf
    https://github.com/cran/penalizedLDA/blob/master/R/PenalizedLDA.R

    Parameters
    ----------

    regs: list, default = None
        The regularization parameters, lambda. Cross-validation is performed
        to select the best value.

    Ks: list, default = None
        The number of discriminant vectors that can be used for classification. 
        Cross-validation is performed to select the best value. The K values 
        must be less than the number of label classes.

    max_iter: int, default = 20
        The number of iterations performed in the minorization procedure used
        to compute the discriminant vectors.

    Attributes
    ----------
    self.K_selected: int
        The value of K selected by cross-validation, and used to train the 
        final model.

    self.reg_selected: float
        The value of the regularization parameter (lambda) selected by cross-
        validation, and used to train the final model.

    self.err_: array, shape = (n_folds, len(regs)) or (n_folds, n_regs, n_Ks)
        The number of errors in each iteration of cross-validation. Three 
        dimensions are used if cross-validation is performed over multiple K
        values.

    self.nonzero_betas: array, shape = (n_folds, len(regs)) or (n_folds, n_regs, 
    n_Ks)
        The number of nonzero parameters in each iteration of cross-validation. 
        Three dimensions are used if cross-validation is performed over multiple 
        K values.

    self.wcsds_: array, shape = (n_parameters,)
        The within-class standard deviation for each parameter in the training
        set.

    self.means_: array, shape = (n_parameters,)
        The mean of each parameter in the training set.

    self.train_err_mean: array, shape = (n_regs,) or (n_regs, n_Ks)
        The mean error across training folds, for different hyperparameter 
        values. This matrix is used to select the hyperparameters. If 
        cross-validation is performed over multiple K values, then a 
        two-dimensional array is used.

    self.X_proj_: array, shape = (n_examples, K_selected)
        The projection of the X values used in cross-validated training, using
        the top K discriminant vectors.

    self.y_: array, shape = (n_examples,)
        The y values used in cross-validated training.
    
    
    '''

    def __init__(self, regs = None, Ks = None, max_iter = 20):
        self.regs = regs
        self.Ks = Ks
        self.max_iter = max_iter
        self.K_selected_ = None
        self.reg_selected_ = None
    def fit(self,X,y,standardized=False,n_folds=5,folds=None):
        # fits the classifier on each fold, and then uses the best hyperparameters to fit
        # the model to all examples.

        if folds is None:
            folds = StratifiedKFold(y,n_folds)
        if y.dtype != int:
            raise ValueError('y must be an array of integers')
        if self.regs is None:
            self.regs = np.logspace(.1,10,5)
        if self.Ks is None: 
            self.Ks = [i for i in range(1,len(set(y)))]
        else:
            if (not (isinstance(self.Ks, list))) or (not any([isinstance(i,int) for i in self.Ks])):
                raise ValueError('Ks must be a list of integers')
            if (np.array(self.Ks) >= len(set(y))).any():
                raise ValueError('K must be less than the number of unique classes')

        # if only one K value needs to be used
        if len(self.Ks) == 1:
            self.K_selected_ = self.Ks[0]
            self.err_ = np.zeros((len(folds),len(self.regs)))
            self.nonzero_betas_ = np.zeros((len(folds),len(self.regs)))

            for (i_fold, (tr,val)) in enumerate(folds):
                print('fold',i_fold+1)
                Xtr = X[tr,:]
                ytr = y[tr]
                Xval = X[val,:]
                yval = y[val]

                # compute mean and within-class SD from training fold, then scale data
                self.wcsds_ = within_class_sds(Xtr,ytr)
                
                if 0 in self.wcsds_:
                    raise ValueError('Some features have 0 within-class standard deviation')
                else:
                    self.means_ = Xtr.mean(axis=0)
                    Xtr = (Xtr - self.means_) / self.wcsds_
                    Xval = (Xval - self.means_) / self.wcsds_
                    
                # the clf classifier is trained with each regularisation parameter
                clf = PenalizedLDA(K = self.K_selected_)
                for (i_reg, reg) in enumerate(self.regs):
                    print('lambda',reg)
                    clf.reg = reg
                    clf.fit(Xtr,ytr,standardized=True)
                    self.nonzero_betas_[i_fold,i_reg] = (clf.discrim_[:,:self.K_selected_]
                                                        ).any(axis=1).sum(axis=0)
                    self.err_[i_fold,i_reg] = clf.error(Xval,yval,standardized=True)
                    
            self.train_err_mean_ = self.err_.mean(axis=0)
            self.reg_selected_ = self.regs[self.train_err_mean_.argmin()]
            
        # if it is necessary to cross-validate over K values
        else:
            self.err_ = np.zeros((len(folds),len(self.regs),len(self.Ks)))
            self.nonzero_betas_ = np.zeros((len(folds),len(self.regs),len(self.Ks)))
            for (i_fold, (tr,val)) in enumerate(folds):
                print('fold',i_fold+1)
                Xtr = X[tr,:]
                ytr = y[tr]
                Xval = X[val,:]
                yval = y[val]

                # scale X using mean and within-class standard deviation from training set
                self.wcsds_ = within_class_sds(Xtr,ytr)
                if 0 in self.wcsds_:
                    raise ValueError('Some features have 0 within-class standard deviation')
                else:
                    self.means_ = Xtr.mean(axis=0)
                    Xtr = (Xtr - self.means_) / self.wcsds_
                    Xval = (Xval - self.means_) / self.wcsds_
                
                # the clf classifier is trained with each combination of hyperparameters
                clf = PenalizedLDA(K=max(self.Ks))
                for (i_reg, reg) in enumerate(self.regs):
                    print('lambda',reg)
                    clf.reg = reg
                    clf.fit(Xtr,ytr,standardized=True)
                    for i_k,k in enumerate(self.Ks):
                        self.err_[i_fold,i_reg,i_k] = clf.error(Xval,yval,k=k,standardized=True)
                        self.nonzero_betas_[i_fold,i_reg,i_k] = (clf.discrim_[:,:k]
                                                                ).any(axis=1).sum(axis=0)

            self.train_err_mean_ = self.err_.mean(axis=0)
            i_best_params = np.unravel_index(self.train_err_mean_.argmin(),self.train_err_mean_.shape)
            self.reg_selected_ = self.regs[i_best_params[0]]
            self.K_selected_ = self.Ks[i_best_params[1]]
            
        # refit model with selected parameters
        selected_model = PenalizedLDA(reg=self.reg_selected_,K=self.K_selected_)
        selected_model.fit(X,y)
        self.discrim_selected_ = selected_model.discrim_
        self.nonzero_ = self.discrim_selected_.any(axis=1).sum(axis=0)
        self.X_proj_ = selected_model.Xtr_proj_
        self.y_ = y
        return(self)

    def predict(self,Xte,standardized=False,k=None):
        # predict y values for the given Xte using the selected model
        if not standardized:
            #set mean to zero and within-class variance to one
            Xte = (Xte-self.means_)/self.wcsds_

        #project to dimensionality k using discriminant
        Xte_projected = Xte.dot(self.discrim_selected_[:,:k])

        #predict class as nearest centroid in projected space
        y_predicted = classify(self.X_proj_[:,:k],self.y_,Xte_projected)

        return(y_predicted)
    def error(self,Xte,yte,k=None, standardized=False):
        # the count of errors in the prediction made using Xte
        if yte.dtype != int:
            raise ValueError('yte must be an array of integers')
        if len(yte)!= len(Xte):
            raise ValueError('Xte and yte must have same number of examples')
        err = (self.predict(Xte,k=k,standardized=standardized)!=yte).sum()
        return(err)
    def __repr__(self):
        return(('Penalized LDA(regs={regs}, Ks={Ks}, max_iter={max_iter}, self.K_selected_='+
               '{K_selected_}, self.reg_selected_={reg_selected_})').format(
                regs=self.regs, Ks=self.Ks, max_iter=self.max_iter, 
                K_selected_=self.K_selected_, reg_selected_=self.reg_selected_))
