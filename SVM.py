
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time
import tensorflow
from tensorflow import keras
from keras import layers,models,Model
from keras.layers import Dropout
from keras.models import load_model
from keras.initializers import HeNormal
from keras.optimizers import Adagrad
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
import os

def unpackTheModel(filename):
    # Check if the file exists
    if os.path.exists(filename):
        model = load_model(filename)
        print("Model loaded successfully.")
        return model
    return None
datagen = ImageDataGenerator(
    rotation_range=40,  
    width_shift_range=0.3,  
    height_shift_range=0.3,  
      shear_range=0.3,  
    zoom_range=0.3,  
    horizontal_flip=True,
    vertical_flip=True,  
    fill_mode='nearest')
def dataAugmentationMLP(x_set):
    """ x_set = x_set.reshape((-1, 32, 32, 3))
     if i==0:
        datagen.fit(x_set)
     x_set = x_set.reshape((-1, 32*32*3))"""
    x_set=x_set**2
    return x_set


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getTheData(filename):
    file_path = os.path.join("cifar-10-batches-py", filename)
    dict=unpickle(file_path)
    lista=list(dict.keys())
    Y =dict[lista[1]]
    X=dict[lista[2]]
    X=X/255
    return np.array(X),np.array(Y)

def gatherTheData():
    x_test,y_test=getTheData("test_batch")
    x_train=np.random.random((0,3072))
    y_train=np.random.random((0))

    for i in range (1,6):
        TEMPX,TEMPY=getTheData("data_batch_"+str(i))
        x_train=np.concatenate([x_train,TEMPX],axis=0)
        y_train=np.concatenate([y_train,TEMPY],axis=0)
    return x_train,y_train,x_test,y_test
def concatenateAug(a):
    b=dataAugmentationMLP(a)
    return np.concatenate([a,b],axis=1)
x_train,y_train,x_test,y_test=gatherTheData()
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)
x_train=concatenateAug(x_train)
x_test=concatenateAug(x_test)

model = models.Sequential()

#Το μοντέλο που παρέχει την απεικόνιση των δεδομένων εισόδου σε 32 διαστάσεις εκπαιδευεται στην
#παρακάτω ρουτίνα
def modelTraining():
    model.add(layers.Flatten(input_shape=(3072*2,)))
    model.add(layers.Dropout(0.2))  
    model.add(layers.Dense(256*4, activation='relu',kernel_initializer=HeNormal()))   
    model.add(layers.Dense(256*2, activation='relu',kernel_initializer=HeNormal()))   
    model.add(layers.Dense(256, activation='relu',kernel_initializer=HeNormal()))  
    model.add(layers.Dense(124, activation='relu',kernel_initializer=HeNormal()))  
    model.add(layers.Dense(32, activation='relu',kernel_initializer=HeNormal()))  
    model.add(layers.Dense(10, activation='softmax',kernel_initializer='glorot_uniform'))
    model.compile(optimizer= Adagrad(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
    h=model.fit(x_train, y_train, batch_size=50,epochs=70,shuffle=True,validation_data=(x_test,y_test))
    model.evaluate(x_test,y_test,batch_size=1)
    model.save('MLP2Final01.keras')
    return model


def i_vs_all(i,y_set):
    """
    i->label
    y_set->list of targets
    μια μέθοδος που μετατρέπει ένα συνολο ετικέτων που προοριζόταν για προβλήμα πολυταξινόμησης σε
    προβλημα δυαδικής ταξινόμησης (θετικη κλαση η i αρνητικες ολες οι υπολοιπες)
    a method that turns i label as 1 and every other class 0 (i-th label is positive class 
    and every other label is negativeclass)
    """
    ivsall=[]
    for target in y_set:
        if(target==i):
            ivsall.append(1)
        else:
            ivsall.append(0)
        
    return ivsall


def acc(target,result):
    """
    η ακρίβεια των προβλεπομενων τιμών (result) και των στόχων (target)
    a function that calculate the accuracy of a prediction set (result)
    """
    count=0
    for i in range(0,target.shape[0]):
        if(target[i]==result[i]):
            count=count+1
    percentage=(count/target.shape[0])*100
    return percentage


def confusionMatrix(i,y,target):
    """
    i->label
    y->prediction set
    target->true y 
    για μία κλαση i υπολογιζονται οι τιμές True Positive(TP), False Negative(FN)
    True Negative (TN) και False Positive(FP). Η κλάση i θεώρειται θετικη και ολες οι υπολοιπες αρνητικες.
    Τέλος, επιστρεφεται μια λιστα απο τις υπολογισμένες τιμές ωστε να γινουν οι καταλληλες πράξεις για να προκτυψουν 
    οι μετρικες αξιολογησης του μοντέλου.

    for every i-label this function calculate the values of True Positive(TP), False Negative(FN)
    True Negative (TN) & False Positive(FP). The i-th class is always the positive class and the rest classes
    are pressumed as one negative class.
    """
    TP=0
    FN=0
    TN=0
    FP=0
    for elementIndex in range(0,target.shape[0]):
        if y[elementIndex]==i:
            if target[elementIndex]==i:
                TP=TP+1
            if target[elementIndex]!=i:
                FN=FN+1
        if y[elementIndex]!=i:
            if target[elementIndex]==i:
                FP=FP+1
            if target[elementIndex]!=i:
                TN=TN+1
    return [TP,FN,FP,TN]

def calculateDistances(svmlist,x_set):
    """
    svmlist->a list of the svm that are being used for the one vs all classification
    x_set->input set
    Αυτη η μεθοδος διατηρεί την  αποσταση των δειγμάτων για κάθε svm σε μια λίστα (distances).

    This method keeps the distance of each example from the SVM line in a list (distances).
    """
    distances = []
    for i in range(0,10):
        distances.append(svmlist[i].decision_function(x_set))
    distances = np.array(distances)
    return distances

def findTheResultsOfOvsA(distances):
    """
    distances->A 10xN array that contains the distance of each example from the svm line.(N: number of examples)

    λαμβανει εναν πινακα διαστάσεων 10ΧΝ με αποστάσεις των Ν δειγμάτων απο
    κάθε svm της λιστας (απο τους 10).H αποστάση σημαίνει σιγουρία στο αποτέλεσμα. Επομένως βρίσκοντας ποιο svm έχει την μεγαλύτερη 
    απόσταση για κάθε δειγμα βρίσκουμε την εκτιμώμενη κλάση του δείγματος.

    Distances can be a measure of confidence in the result of each svm. So for each input the svm that has the biggest distance from it
    can be used to predict its class.


    

    """
    distances=np.array(distances)
    results=np.argmax(distances, axis=0)
    model_precision=acc(y_test,results)
    return results



def makePositiveClass(y_train):
    """
    Με βαση τα δεδομένα εκπαιδευσης για κάθε κλάση i
    δημιουργειται μια λίστα στόχων που θεωρουν την i ως θετικη και τις υπολοιπες αρνητικες.
    Στην συνέχεια κρατάμε τις λίστες κάθε i κλασης στην λίστα ytrain_i_vs_all.

    for each i-label we create a binary set where i =1 and all the other labels=0, then those sets are stored in a list 
    (ytrain_i_vs_all)
    
    """
    ytrain_i_vs_all=[]
    for i in range(0,10):
        ytrain_i_vs_all.append(i_vs_all(i,y_train))
    return ytrain_i_vs_all
def oneVsAllSVM(train_set,y_train):
    """
    train_set->training input
    y_train->training output

    δημιουργια δεκα μοντελων SVM με σκοπο την υλοποίηση One Vs All.
    Στην συνεχεία για κάθε SVM γινεται ενα grid search ωστε να βρεθουν οι καλυτερες παραμετροι για αυτο και προστήθεται σε μια λιστα
    SVM με το αναγνωριστικο svmlist.
    Τέλος για την κατηγοριοποιήση στο test set, σε καθε svm βρισκουμε την αποσταση που εχουν τα δειγματα ελεγχου απο την
    εκαστοτε κατηγοριοποιητικη επιφάνεια. Κάθε δείγμα κατατάσετε στην κλάση σύμφωνα με την θετική κλαση του
    svm που, το δείγμα, είχε την μεγαλύτερη απόσταση. 

    We create 10 svm for the purpose of One Vs All classification.
    For each svm we are using grid search to find the best parameters, then its stored in a list svmlist.
    Finally, each input is classified as the positive class of the svm that has the biggest distance from it. 

    """
    ytrain_i_vs_all=makePositiveClass(y_train)
    svmlist=[]
    for i in range(0,10):
         svm_classifier =SVC()
         param_grid = {'C': [1,0.01,0.1,10,100], 'gamma': [0.01, 0.001, 0.0001,0.00001],"degree":[2,4,5] , 'kernel': ['rbf','poly']}
         grid_search = GridSearchCV(svm_classifier, param_grid, cv=2)
         grid_search.fit(train_set,ytrain_i_vs_all[i])
         print("classifier "+str(i)+" has best parameters: "+str(grid_search.best_params_))
         svmlist.append(grid_search.best_estimator_)
    distances=calculateDistances(svmlist,test_set)
    results=findTheResultsOfOvsA(distances)

    distances_train=calculateDistances(svmlist,train_set)
    results_train=findTheResultsOfOvsA(distances_train)
    print("Η ακριβεία του μοντέλου one vs all, μετα την μερική εκπαίδευση, στο σύνολο ελέγχου είναι "+str(acc(y_test,results))+" %.")
    print("Η ακριβεία του μοντέλου one vs all, μετα την μερική εκπαίδευση, στο σύνολο εκπαίδευσης είναι "+str(acc(y_train,results_train))+" %.")
    print("\n")
    return svmlist,results

def SVMinBatches(train_set,y_train):
    """
    train_set->training input
    y_train->training output

    μια βελτίωση ως προς την αποψη μνήμης. Χωριζουμε το αρχικο dataset σε μικροτέρα ξένα σύνολα (συγκεκριμένα 6) και σειριακά λυνουμε το 
    το πρόβλημα της ευρεσης των διανυσμάτων υποστήριξης. Αφου τα βρούμε τα προσθέτουμε στο έπομενο συνολο και ξαναλυνουμε το προβλημα.

    An improvment in terms of memory usage. We split the dataset in smaller disjoint sets (6 in particular). We solve the problem of finding 
    the support vectors for every set sequentially and after every step the S.V. are added to the next set.
    """
    x1,x2,y1,y2=train_test_split(train_set,y_train,test_size=0.5)
    x11,x12,y11,y12=train_test_split(x1,y1,test_size=0.5)
    x111,x112,y111,y112=train_test_split(x11,y11,test_size=0.5)
    x121,x122,y121,y122=train_test_split(x12,y12,test_size=0.5)

    x21,x22,y21,y22=train_test_split(x2,y2,test_size=0.5)
    x211,x212,y211,y212=train_test_split(x21,y21,test_size=0.5)
    x221,x222,y221,y222=train_test_split(x22,y22,test_size=0.5)

    xset=list([x111,x112,x121,x122,x211,x212,x221,x222])
    yset=list([y111,y112,y121,y122,y211,y212,y221,y222])
    max=0
    
    modelC=SVC(decision_function_shape='ovo',kernel='rbf',C=0.1,gamma=0.001)
    numberOfSets=len(xset)
    a=time.time()
    for i in range(numberOfSets):
        modelC.fit(xset[i], yset[i])
        support_vectors = modelC.support_vectors_
        support_vector_positions_in_trainset = list(modelC.support_)
        if(i<numberOfSets-1):
            xset[i+1] = np.concatenate([xset[i+1],support_vectors])
            yset[i + 1] = np.append(yset[i+1], yset[i][support_vector_positions_in_trainset])
        if(max<xset[i].shape[0]):
            max=xset[i].shape[0]
    b=time.time()
    print("Ο τεμαχισμός απαιτεί "+str(b-a)+" δευτερολεπτα.")
    print("Το μέγεθος του συνόλου που το SVM εκπαιδεύτηκε στο τελευταίο στάδιο ήταν: "+str(xset[i].shape[0]))
    print("Το μεγαλύτερο συνολο που το SVM εκαπαιδευτηκε σε ένα σταδιο είχε πλήθος: "+str(max))
    print("Ακρίβεια με την μέθοδο του τεμαχισμού στο σύνολο ελέγχου "+str(acc(y_test,modelC.predict(test_set))))
    print("Ακρίβεια με την μέθοδο του τεμαχισμού στο σύνολο εκπαίδευσης "+str(acc(y_train,modelC.predict(train_set))))
    
    print("\n")
    return modelC 

def simpleSVMmodel(train_set,y_train,test_set,y_test,kernel,c,g,d):
    """
    train_set->input training set
    y_train->output training set
    test_set->input testing set
    y_test->output testing set
    kernel->the kernel function that will be used
    c->error tolerance
    g->gamma value of gauss function (rbf kernel)
    d->degree of a poly function
    εκπαιδευση και αξιολόγηση ένος απλού μοντέλου SVM
    εμφανιζει τις παραμέτρους του μοντέλουυ και την ακρίβεια του
    συν τον χρόνο που απαιτεί η εκπαίδευση
    επιστρέφει το μοντέλο που εκπαιδευτηκε και την ακρίβεια στο συνολο ελέγχου

    Training and validating a simple svm model given by the sklearn library.
    Info about the hyperparameters and the time required for the training of the models are being printed in
    the user screen.

     
    """
    flag=0
    print("To SVM με συνάρτηση πηρύνα : "+kernel,end=",")
    if(kernel == 'rbf'):
        print(" με C= "+str(c)+" και gamma= "+str(g)+".")
        modelSVC= SVC(decision_function_shape='ovo',kernel=kernel,C=c,gamma=g)
        flag=1
    if(kernel=="linear"):
         print(" με C :"+str(c)+".")
         modelSVC= SVC(decision_function_shape='ovo',kernel=kernel,C=c)
         flag=1
    if(kernel=="poly"):
        print(" με C= "+str(c)+" και degree= "+str(d)+".")
        modelSVC= SVC(decision_function_shape='ovo',kernel=kernel,C=c,degree=d)
        flag=1

    if(flag==0):
        print("Σφάλμα! Δόθηκε λάθος όνομα πυρήνα.")
        return None
    
    start_time = time.time()
    modelSVC.fit(train_set, y_train)
    end_time = time.time()
    elapsed_time=(end_time-start_time)
    print("Ο χρονος που απαιτήθηκε για την εκπαιδευση του μοντέλου: "+str(elapsed_time)+" δευτερόλεπτα.")
    precision_test=modelSVC.score(test_set, y_test)
    precision_train=modelSVC.score(train_set, y_train)

    print("Ποσοστό ακρίβειας στα δεδομένα εκπαίδευσης: "+str(precision_train)+"\nκαι στα δεδομένα ελέγχου: " + str(precision_test))
    print("\n")
    return modelSVC,precision_test
    



MLP1_=unpackTheModel("MLP2Final01.keras")
if (MLP1_==None):
     MLP1_=modelTraining()


def projectTheDataTo32D():
    """
    Η μείωση των δειγμάτων εισόδου σε 32 με βάση το MLP μοντέλο που εκπαιδευεται με την συναρτηση modelTraining()

    Decreasing the dimention of the input matrix from 50000x3072 to only 500000x32 using the MLP model
    """
    hidden_layer_model = Model(inputs=MLP1_.input, outputs=MLP1_.layers[6].output)
    train_set=hidden_layer_model.predict(x_train)
    test_set=hidden_layer_model.predict(x_test)
    return train_set,test_set
    
#ξανακάνω τους στόχους βαθμωτούς (απο one-hot-encoding)
y_train=np.argmax(y_train, axis=1)
y_test=np.argmax(y_test, axis=1)

train_set,test_set=projectTheDataTo32D()  #6114 diamentions to 32


g=0
d=0

max=0
#εκπαιδευσή των απλών svm με δίαφορες παραμέτρους και διατηρούμε τον καλύτερο μοντέλο για κάθε μοντέλο.
#training the simple SVM models with diffrent parameters and keeping track of the best model of each kernel
for c in [0.1,1,10,100]: 
    model,precision=simpleSVMmodel(train_set,y_train,test_set,y_test,"linear",c,g,d)
    if(precision>max):
         max=precision
         bestLinear=model
max=0 
for c in [100,10,1,0.1,0.01]:       
    for d in [3,5,7,9]: 
        model,precision=simpleSVMmodel(train_set,y_train,test_set,y_test,"poly",c,g,d)
        if(precision>max):
            max=precision
            bestPoly=model
max=0
for c in [100,1,0.1]:
    for g in [0.1,0.001,0.0001,0.00001,0.000001,0.0000001]: 
        model,precision=simpleSVMmodel(train_set,y_train,test_set,y_test,"rbf",c,g,d)
        if(precision>max):
            max=precision
            bestRBF=model




       
#το x2 είναι ενα ομαλοποίημενο υποσύνολο του αρχικού συνόλου και θα χρησιμοποιήθει για την επιλογή μοντέλου
x1,x2,y1,y2=train_test_split(train_set,y_train,test_size=0.8,random_state=42)
svmlist,onevsallprecision=oneVsAllSVM(x2,y2)
y_train_i_vs_all,temp=makePositiveClass(y_train)
timestart=time.time()
#εκπαίδευση του μοντέλου σε όλο το συνολο εκπαιδευσής
for i in range(0,10):
    svmlist[i].fit(train_set,y_train_i_vs_all[i])
timeend=time.time()
distances=calculateDistances(svmlist,test_set)
pred_of_OneVsAll=findTheResultsOfOvsA(distances)

traindistances=calculateDistances(svmlist,train_set)
print("Ακρίβεια του one vs all στο σύνολο εκπαίδευσης ειναι: "+str(acc(y_train,findTheResultsOfOvsA(traindistances))))
print("Αποτελεσμα μετα απο την εκπαιδευση με όλα τα παραδείγματα του μοντέλου ένα εναντίον όλων στο σύνολο ελεγχου :"+str(acc(y_test,pred_of_OneVsAll)))
print("Η εκπαίδευση απαίτησε "+str(timeend-timestart)+" δευτερόλεπτα.")
print("")

modelChunking=SVMinBatches(train_set,y_train)
modelChunkingPred= modelChunking.predict(test_set)  

bestLinearPred= bestLinear.predict(test_set)
bestPolyPred=bestPoly.predict(test_set)
bestRBFPred=bestRBF.predict(test_set)


confusionOfChunk=[]
confusionOfBestPoly=[]
confusionOfBestRBF=[]
confusionOfBestLinear=[]
confusionOfOneVsAll=[]

#αξιολογηση 
#evaluation
# for i in range(0,10):
#     confusionOfChunk.append(confusionMatrix(i,modelChunkingPred,y_test))
#     confusionOfOneVsAll.append(confusionMatrix(i,pred_of_OneVsAll,y_test))
#     confusionOfBestPoly.append(confusionMatrix(i,bestPolyPred,y_test))
#     confusionOfBestRBF.append(confusionMatrix(i,bestRBFPred,y_test))
#     confusionOfBestLinear.append(confusionMatrix(i,bestLinearPred,y_test))
def findCorrectAndIncorrectAnswers(modelTitle,y_target,y_pred):
    """
    Η συνάρτηση εκτυπώνει παραδείγματα που ταξινόμηθηκαν σωστα και λάθος.(5 παραδειγματα για κάθε κατηγορια)
    
    a function that shows examples that were classified as the right/wrong class (5 examples of each)
    """
    print("Στο μοντέλο "+modelTitle)
    listOfErrors=[]
    listOfCorrects=[]
    limit=5
    limitErr=5
    i=0
    flag1=False
    flag2=False
    while (not(flag1) or not(flag2)) or i<y_target.shape[0]:
        if(limit>0 and y_pred[i]==y_target[i]):
            limit=limit-1
            flag1=True
            listOfCorrects.append(i)
        if(limitErr>0 and y_pred[i]!=y_target[i]):
            limitErr=limitErr-1
            flag2=True
            listOfErrors.append([i,y_pred[i],y_target[i]])

        i=i+1

    print("Μερικά παραδείγματα του συνόλου ελέγχου που ταξινομήθηκαν εσφαλμένα είναι τα εξής")
    for x in listOfErrors:
        print("To δείγμα "+str(x[0])+" ταξινομήθηκε ως "+str(x[1])+", ενώ ήταν "+str(x[2]))
    print("Μερικά από τα παραδείγματα του συνόλου ελέγχου που ταξινομήθηκαν σωστά είναι τα εξής")
    for x in listOfCorrects:
        print("To δείγμα "+str(x)+" ταξινομήθηκε ως "+str(y_pred[x]))
def confusionMatrixPresentantion(modelTitle,confusionClass,y_pred,y_true):
    """
    modelTitle->model name
    confusionClass->
    y_pred->prediction of the model
    y_true->expected output
    Για το μοντέλο που δόθηκε, υπολόγιζει μετρικές (accuracy,precision,F1,Recall) με βάσει τον πινακα συγχυσης του μοντέλου (micro analysis)

    Calculating diffrent metrics (accuracy,precision,F1,Recall) for the given model based in the model's confusion Matrix
    (micro analysis)
    """
    print("οι πινακες συγχυσης του μοντέλου "+modelTitle+" για κάθε κλάση είναι oι παρακάτω :")
    final_con=[[0,0],[0,0]]
    
    sumRec=0
    sumPrec=0
    sumAccuracy=0
    sumF1=0
    for i in range (0,10):
        print("Ο πινακας συγχυσης με την κλάση "+str(i)+" ως θετικη. ")
        final_con[0][0]=confusionClass[i][0]
        final_con[0][1]=confusionClass[i][2]
        final_con[1][0]=confusionClass[i][1]
        final_con[1][1]=confusionClass[i][3]
        print(str(final_con[0]))
        print(str(final_con[1]))
        print("")
        TP=final_con[0][0]
        FP=final_con[0][1]
        FN=final_con[1][0]
        TN=final_con[1][1]
        sumPrec+=TP/(TP+FP)
        sumRec+=TP/(TP+FN)
        sumAccuracy+=(TP+TN)/(TP+FP+TN+FN)
        sumF1+=(2*TP)/(2*TP+FN+FP)
    
    

    print("Ακριβεια μοντέλου: "+str(sumPrec/10))
    print("Ορθότητα μοντέλου: "+str(sumAccuracy/10))
    print("Recall μοντέλου: "+str(sumRec/10))
    print("F1 μοντέλου: "+str(sumF1/10))
    findCorrectAndIncorrectAnswers(modelTitle,y_true,y_pred)
    print("")

chunkConfusionList=[]
OneVsAllConfusionList=[]
bestLinearConfusionList=[]
bestRBFConfusionList=[]
bestPolyConfusion=[]
for i in range(0,10):
    chunkConfusionList.append(confusionMatrix(i,modelChunkingPred,y_test))
    OneVsAllConfusionList.append(confusionMatrix(i,pred_of_OneVsAll,y_test))
    bestLinearConfusionList.append(confusionMatrix(i,bestLinearPred,y_test))
    bestRBFConfusionList.append(confusionMatrix(i,bestRBFPred,y_test))
    bestPolyConfusion.append(confusionMatrix(i,bestPolyPred,y_test))

confusionMatrixPresentantion("Μοντέλο Τεμαχισμού",chunkConfusionList,y_test,modelChunkingPred)
confusionMatrixPresentantion("Μοντέλο One vs All ",OneVsAllConfusionList,y_test,pred_of_OneVsAll)
confusionMatrixPresentantion("Linear",bestLinearConfusionList,y_test,bestLinearPred)
confusionMatrixPresentantion("Poly",bestPolyConfusion,y_test,bestPolyPred)
confusionMatrixPresentantion("RBF",bestRBFConfusionList,y_test,bestRBFPred)


