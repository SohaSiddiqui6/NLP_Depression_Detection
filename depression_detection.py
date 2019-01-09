import pandas as pd 
import re
import gensim 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import warnings
from  sklearn.metrics import accuracy_score
from sklearn import preprocessing
warnings.filterwarnings("ignore")

data = pd.read_csv("Depressiondata.csv")
data.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
data.drop(["a"], axis=1, inplace=True)

def array_cleaner(array):
  X = []
  for sentence in array:
    clean_sentence = ''
    review = re.sub('[^a-zA-Z]', ' ', str(sentence))
    review = review.lower()
    words = review.split(' ')
    for word in words:
        clean_sentence = clean_sentence +' '+ word
    X.append(clean_sentence)
  return X


import math
for i in range(0,36513):
    if data["Depression"][i]== math.isnan(i):
           data["Depression"][i]= "True"
                
X = data.iloc[:,0]
Y = data.iloc[:,1].astype("str")

train_X, X_test, train_Y, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
    
train_X = array_cleaner(train_X)
X_test = array_cleaner(X_test)


num_features = 300  
min_word_count = 1 
num_workers = 4     
context = 10        
downsampling = 1e-3 

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
model = word2vec.Word2Vec(train_X,\
                          workers=num_workers,\
                          size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling)
model.init_sims(replace=True)
model_name = "Depression_Analysis"
model.save(model_name)


def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list 
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec
# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features).T
        counter = counter+1
        
    return reviewFeatureVecs
    
trainDataVecs = getAvgFeatureVecs(train_X, model, num_features)
testDataVecs = getAvgFeatureVecs(X_test, model, num_features)
forest = RandomForestClassifier(n_estimators = 5)
#classifier = SGDClassifier()  
forest = forest.fit(trainDataVecs, train_Y)
#classifier.fit(trainDataVecs,train_Y)
#testdata=['feel lost','sick of this life','pathetic world']
#testDataVecs = getAvgFeatureVecs(testdata, model, num_features)
result = forest.predict(testDataVecs)
print(result)
#y_pred = classifier.predict(testDataVecs)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
accuracy=accuracy_score(y_pred,y_test)
print(accuracy*100)
