from sklearn.datasets import load_iris
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

print ("hola")

datosIris = load_iris()

print ("El numero de instancias es: " + str(len(datosIris.target)))
print ("El numero de features es: " + str(len(datosIris.data[0,:])))


# dividir en entrenamiento y test



x = datosIris.data
y = datosIris.target


rs = ShuffleSplit(n_splits=10,test_size=.25)

scores = []

for train_index, test_index in rs.split(x):
    #print("TRAIN:", train_index, "TEST:",test_index)
    
    xtraining = x[train_index,:]
    xtest = x[test_index,:]
    ytraining = y[train_index]
    ytest = y[test_index]
    
    myclf = KNeighborsClassifier(n_neighbors=3)
    
    myclf.fit(xtraining,ytraining)
    
    ypred = myclf.predict(xtest)
    
    scores.append(float(sum(ypred==ytest))/len(ytest))
    
    sum(scores)/10