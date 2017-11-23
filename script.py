# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read data into dataset variable
mushrooms=pd.read_csv("../input/mushrooms.csv")
mushrooms.head()

from sklearn.preprocessing import LabelEncoder

for c in mushrooms.columns:
    mushrooms[c]=mushrooms[c].fillna(-1)
    if mushrooms[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(mushrooms[c].values))
        mushrooms[c] = lbl.transform(list(mushrooms[c].values))
        
print(mushrooms.describe().T)
mushrooms


for k in mushrooms.columns:
    mushrooms[k]=mushrooms[k].fillna(-1)
    if mushrooms[k].dtype=='object':
        lbl = LabelEncoder()
        lbl.fit(list(mushrooms[k].values))
        mushrooms[k]=lbl.transform(list(mushrooms[k].values))
          
          
print(mushrooms.describe().T)
mushrooms


print("I am going to print some Stuff so Be Careful")
n = 0

print ("Pattern A")
for x in range (0,11):
    n = n + 1
    for a in range (0, n-1):
        print ('*', end = '')
    print()
print ('')
print ("Pattern B")
for b in range (0,11):
    n = n - 1
    for d in range (0, n+1):
        print ('*', end = '')
    print()
print ('')

print ("Pattern C")
for e in range (11,0,-1):
    print((11-e) * ' ' + e * '*')

print ('')
print ("Pattern D")
for g in range (11,0,-1):
    print(g * ' ' + (11-g) * '*')

classesnames= mushrooms.groupby('class').mean()
print(classesnames.head().T)


def dddraw(X_reduced,name):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired)
    titel="First three directions of "+name 
    ax.set_title(titel)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()
    
    
from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.cluster import KMeans,Birch
import statsmodels.formula.api as sm
from scipy import linalg
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
import matplotlib.pyplot as plt

n_col=20
X = mushrooms.drop(['class'],axis=1) 

def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)

Y=mushrooms['class']
X=X.fillna(value=0)       # those ? converted to NAN are bothering me abit...        
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)


names = [
         'PCA',
         'FastICA',
         'Gauss',
         'KMeans',
         #'SparsePCA',
         #'SparseRP',
         'Birch',
         'NMF',    
         'LatentDietrich',    
        ]

classifiers = [
    
    PCA(n_components=n_col),
    FastICA(n_components=n_col),
    GaussianRandomProjection(n_components=3),
    KMeans(n_clusters=24),
    #SparsePCA(n_components=n_col),
    #SparseRandomProjection(n_components=n_col, dense_output=True),
    Birch(branching_factor=10, n_clusters=12, threshold=0.5),
    NMF(n_components=n_col),    
    LatentDirichletAllocation(n_topics=n_col),
    
]
correction= [1,1,0,0,0,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    Xr=clf.fit_transform(X,Y)
    dddraw(Xr,name)
    res = sm.OLS(Y,Xr).fit()
    #print(res.summary())  # show OLS regression
    #print(res.predict(Xr).round()+correct)  #show OLS prediction
    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction

    #print('Ypredict *log_sec',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction
    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y))    
    
    
    