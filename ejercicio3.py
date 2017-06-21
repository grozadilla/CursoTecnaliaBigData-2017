from sklearn.datasets import load_iris

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import numpy as np



datosIris = load_iris()

print ("El numero de instancias es: " + str(len(datosIris.target)))
print ("El numero de features es: " + str(len(datosIris.data[0,:])))


Xraw = datosIris.data
y = datosIris.target

myMinMaxScaler = MinMaxScaler()
Xsc = myMinMaxScaler.fit_transform(Xraw)

mypca = PCA(n_components=2)
X = mypca.fit_transform(Xsc)

#rejilla_parametros = {"n_neighbors":[3,5,7,9,11,13], "weights": ["uniform","distance"]}
rejilla_parametros = {"n_neighbors":range(1,15), "weights": ["uniform","distance"], "p":[2,3]}

myclf = GaussianNB()
#mygridsearchcv = GridSearchCV(myclf,rejilla_parametros,cv=10)
#mygridsearchcv.fit(X,y)


#myclf = mygridsearchcv.best_estimator_
myclf.fit(X,y)


NPOINTS =100

minimos = np.min(X,axis=0)
maximos = np.max(X,axis=0)

vector_x0 = np.linspace(minimos[0],maximos[0],100)
vector_x1 = np.linspace(minimos[1],maximos[1],100)

xx0,xx1 = np.meshgrid(vector_x0,vector_x1)

predictionsRejilla = np.zeros((NPOINTS,NPOINTS))

for i in range(NPOINTS):
    for j in range(NPOINTS):
        predictionsRejilla[i,j]=myclf.predict([xx0[i,j],xx1[i,j]])
        
from matplotlib import pyplot as plt

plt.figure()
plt.matshow(predictionsRejilla)

markers = ['o','s','x']
colors = ['b', 'g', 'r']

plt.figure()

for k in range(len(X)):
    plt.scatter(X[k,0], X[k,1], c=colors[y[k]], marker=markers[y[k]])
    
plt.show()





