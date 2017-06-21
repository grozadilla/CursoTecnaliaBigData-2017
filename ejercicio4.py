from sklearn.datasets import load_digits

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np



datosIris = load_digits()

print ("El numero de instancias es: " + str(len(datosIris.target)))
print ("El numero de features es: " + str(len(datosIris.data[0,:])))

Xraw = datosIris.data
y = datosIris.target

myMinMaxScaler = MinMaxScaler()
Xsc = myMinMaxScaler.fit_transform(Xraw)

mypca = PCA(n_components=10)
X = mypca.fit_transform(Xsc)

myclf = KNeighborsClassifier()
scores = cross_val_score(myclf,X,y,cv=10)


print (scores)
print (np.mean(scores))
print (np.std(scores))







