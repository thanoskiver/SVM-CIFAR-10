import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

#

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def dataModification(filename):
    dataFile=unpickle("cifar-10-batches-py\\"+filename)
    alist=list(dataFile.keys())
    x=dataFile.get(alist[2])
    y=dataFile.get(alist[1])
    return x,y
#αρχικοποιηση των κατηγοριοποιητων
knn1_classifier = KNeighborsClassifier(n_neighbors=1)
knn3_classifier = KNeighborsClassifier(n_neighbors=3)
nearest_centroid_classifier = NearestCentroid()

x_test,y_test=dataModification("test_batch")
listOfBatches=["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
arrayX=np.random.random((0,3072))
arrayY=np.random.random((0))

for filename in listOfBatches:
    dataFile=unpickle("cifar-10-batches-py\\"+filename)
    alist=list(dataFile.keys())
    Y=dataFile.get(alist[1])
    X=dataFile.get(alist[2])
    X=np.array(X)
    Y=np.array(Y)   
    x_train, y_train= X,Y
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    
    arrayX=np.concatenate([arrayX,X_train_scaled],axis=0)
    arrayY=np.concatenate([arrayY,y_train],axis=0)

    


knn1_classifier.fit(arrayX, arrayY)
knn3_classifier.fit(arrayX, arrayY)
accuracy1= knn1_classifier.score(X_test_scaled, y_test)
accuracy2=knn3_classifier.score(X_test_scaled, y_test)

print(f'Ακρίβεια ταξινομητή KNN-1: {accuracy1}, ακρίβεια ταξινομητή ΚΝΝ-3: {accuracy2}')
nearest_centroid_classifier.fit(arrayX, arrayY)
accuracy = nearest_centroid_classifier.score(X_test_scaled, y_test)
print(f'Ακρίβεια ταξινομητή Πλησιέστερου Κέντρου: {accuracy}')
  





