from sklearn.datasets import load_iris
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

print ("hola")

datosIris = load_iris()

print ("El numero de instancias es: " + str(len(datosIris.target)))
print ("El numero de features es: " + str(len(datosIris.data[0,:])))


x = datosIris.data
y = datosIris.target

rejilla_parametros = {"n_neighbors":[3,5,7,9,11,13], "weights": ["uniform","distance"]}

from sklearn.model_selection import GridSearchCV

myclf = KNeighborsClassifier()
mygridsearchcv = GridSearchCV(myclf,rejilla_parametros, )


