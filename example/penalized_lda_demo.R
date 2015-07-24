library(penalizedLDA)
df = read.csv('/home/ryan/ml/dm/sun.csv',row.names=1)
y = df$label
X = df[,2:ncol(df)]
X[117,43073] =mean(as.numeric(X[,43073]),na.rm=T)
X = as.matrix(X)
y = as.numeric(y)

setwd(dirname(sys.frame(1)$ofile))
folds = read.csv('folds.csv',row.names=1)
n_cv = 5
n_folds = 10
folds_val = list()
folds_tr = list()
folds_main = list()
for (i in 1:n_folds){
  folds_val[[i]] = folds[,(1+(i-1)*n_cv):(i*n_cv)]
  folds_main[[i]] = unlist(apply(folds_val[[i]],1,any))
  folds_tr[[i]] = (!folds_val[[i]]) & folds_main[[i]]
  folds_val[[i]] = as.list(folds_val[[i]])
  folds_tr[[i]] = as.list(as.data.frame(folds_tr[[i]]))
}

# to test PenalizedLDA, we run the analysis, training on the 75% of data, and test on the other 25%
i_tr = folds_main[[1]]
i_te = !folds_main[[1]]
out = PenalizedLDA(X[i_tr,],y[i_tr],xte=X[i_te,],lambda=.006,K=2)
print('errors')
print(sum(out$ypred[,2]!=y[i_te]))
print('nonzeros')
print(sum(apply(out$discrim,1,sum)!=0))
