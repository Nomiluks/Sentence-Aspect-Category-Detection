from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import sklearn
import math, scipy
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from src.tools import print_confusion_matrix
from django.utils.encoding import smart_str
from bs4 import BeautifulSoup
import xml.etree.ElementTree
import tensorflow as tf
import numpy as np
import pandas as pd  
import os
import re
import random
import glob


categories = np.array(['AMBIENCE#GENERAL','DRINKS#PRICES','DRINKS#QUALITY','DRINKS#STYLE_OPTIONS','FOOD#PRICES','FOOD#QUALITY','FOOD#STYLE_OPTIONS','LOCATION#GENERAL','RESTAURANT#GENERAL','RESTAURANT#MISCELLANEOUS','RESTAURANT#PRICES','SERVICE#GENERAL'])

#generating Lableled Training Dataset for Sentences
def getTraining_setFile(message, inputFile, out):   
    outputFile = out
    reviews = xml.etree.ElementTree.parse(inputFile).getroot()
    count = 0
    try:
        os.remove(outputFile)
    except OSError:
        pass
    
    with open(outputFile, "a") as myfile:
        for review in reviews:
            for sentences in review:
                for sentence in sentences:
                    if (sentence.get('OutOfScope') != "TRUE"):
                        for data in sentence:
                            if data.tag == "text":
                                s = data.text
                                if (len(sentence) == 1):
                                    cat = "NONE"
                                    s += "\t"+cat
                                    s = smart_str(s+"\n") 
                                    #myfile.write(s)
                            elif data.tag == "Opinions":
                                opinions = data
                                if (len(opinions) == 0):
                                    cat = "NONE"
                                    s += "\t"+cat
                                    string = smart_str(s+"\n") 
                                    #myfile.write(string)
                                    break
                                mul=0
                                for opinion in opinions:
                                    #string = s+"\t"+cat
                                    #string = smart_str(string+"\n") 
                                    #myfile.write(string)
                                    #count = count+1
                                    
                                    #for multiple categories
                                    cat = opinion.get('category')
                                    if (mul == 0):
                                        s   += "\t"+cat
                                        mul +=1
                                    else:
                                        s   += "*"+cat
                                
                                string = s
                                string = smart_str(string+"\n") 
                                myfile.write(string)
                                count = count+1
                    else:
                         for data in sentence:
                            if data.tag == "text":
                                s = data.text
                                cat = "OutOfScope"
                                s += "\t"+cat
                                s = smart_str(s+"\n") 
                                #myfile.write(s)
    print ( message,count )

#Extracting Sentences from Training Datasets
def getSentencesFile(inputFile, out): 
    outputFile = out
    reviews = xml.etree.ElementTree.parse(inputFile).getroot()
    count = 0
    try:
        os.remove(outputFile)
    except OSError:
        pass
    
    with open(outputFile, "a") as myfile:
        for review in reviews:
            for sentences in review:
                for sentence in sentences:
                        for data in sentence:
                            if data.tag == "text":
                                s = data.text
                                s = smart_str(s+"\n") 
                                myfile.write(s)
                                count = count+1
                                
    print ( "Total Unique Sentences: ",count )

    
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def review_to_wordlist( review, remove_stopwords=True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    
    # 1. Remove HTML
    #review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = review
    #  
    # 2. Remove non-letters
    #review_text = re.sub("[^a-zA-Z]"," ", review_text)
    review_text  = clean_str(review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

def add_unknown_words(k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    return np.random.uniform(-0.25,0.25,k)  
            
def getWordVecs(words):
    vecs = []
    for word in words:
        #print ( word,"a" )
        word = word.replace('\n', '')
        #print ( word,"a")
        try:
            vecs.append(model[word].reshape((1,300)))
            #print ( "found!",word)
        except KeyError:
            print ( "not found!",word)
            continue
    vecs = np.concatenate(vecs)
    return np.array(vecs, dtype='float') #TSNE expects float type values

###**********combine datasets for word2vec training************###
def word2vecInput(train,test,gold,yelp, shuffle=True, remove_stopwords=True):    
    #initialized train and gold test sentences
    goldSet  = gold["review"].tolist()
    trainSet = train["review"].tolist()
    testSet  = test["review"].tolist()
    
    trainSentences = yelp+trainSet#+testSet+goldSet
    print ( "Total Train Sentences: ",len(trainSentences))
    print ( "Total Gold Test Sentences: ", len(goldSet))
    
    #cleaning sentences    
    trainingSentences = [review_to_wordlist(review,remove_stopwords) for review in trainSentences]
    goldSentences     = [review_to_wordlist(review,remove_stopwords) for review in goldSet]
    #vocablary = trainingSentences+goldSentences
    #print ( "Total Vocablary of Sentences: ", len(vocablary))
    
    #Shuffle training sentences Sentences             
    if (shuffle):
        random.shuffle(trainingSentences)
        
    return trainingSentences#, vocablary

def tokenizerCleaner(yelp, remove_stopwords=True):
    print ("Total Train Sentences: ",len(yelp))    
    yelp = [review_to_wordlist(review,remove_stopwords) for review in yelp]  
    return yelp

def oneHotVectors(labels):
    labels = pd.Series(labels)
    labels = pd.get_dummies(labels)
    labels.to_csv("Labels.csv")
    labels = labels.values
    return labels

def multiHotVectors(categories, name):
    #http://stackoverflow.com/questions/18889588/create-dummies-from-column-with-multiple-values-in-pandas
    dummies = pd.get_dummies(categories['category'])
    atom_col = [c for c in dummies.columns if '*' not in c]
    for col in atom_col:
        categories[col] = dummies[[c for c in dummies.columns if col in c]].sum(axis=1)

    # categories.to_csv(name+".csv")   
    return categories

def categoriesToLabels(labels):
    labels = pd.Series(labels)
    labels = pd.get_dummies(labels)
    labels = labels.values #copying the vectors for each category [1...13] length
    item_index = np.where(labels[:]==1)
    labels = item_index[1]
    return labels

def nextBatch(X_train, y_train, batch_size):
    sample = random.sample(zip(X_train,y_train), batch_size)
    sample = zip(*sample)
    return sample[0], sample[1]

def weight_variable(fan_in, fan_out, filename, boolean=False):   
    initial=0
    if (boolean):
        stddev = np.sqrt(2.0/fan_in)
        initial  = tf.random.normal([fan_in,fan_out], stddev=stddev)
    else:
        initial  = np.loadtxt(filename).astype(np.float32)
        #print ( initial.shape)
    return tf.Variable(initial)

def resetModel():
    files = glob.glob('params/*')
    for f in files:
        os.remove(f)
    
def bias_variable(shape, filename, boolean=False):
    initial=0
    if (boolean):
        initial = tf.constant(0.1, shape=shape)
    else:
        initial  = np.loadtxt(filename).astype(np.float32) 
        #print ( initial.shape)
    return tf.Variable(initial)

def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):
    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    
def confusionMatrix(text,Labels,y_pred, not_partial):
    print ( Labels[7])
    y_actu = np.where(Labels[:]==1)[1]
    print ( y_actu[7])
    df = print_confusion_matrix(y_pred,y_actu)
    print ( "\n",df)
    # print ( plt.imshow(df.as_matrix()))
    if not_partial:
        print ( "\n",classification_report(y_actu, y_pred))
    print ( "\n\t------------------------------------------------------\n")

def normalize(probs):
    prob_factor = 1 / np.sum(probs)
    return [prob_factor * p for p in probs]
    
def do_eval(sent, point, message, sess, correct_prediction, accuracy, pred, X_, y_,x,y, single=False, threshold=0.80, debug=False):
        
    #initializations
    output=[]
    thres = threshold
    correct = False
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    
    #measure Accuracy overall
    if single == "lock":
        #check scores of Nueral Network
        predictions = sess.run([correct_prediction], feed_dict={x: X_, y: y_})
        prediction  = tf.argmax(pred,1)
        labels = prediction.eval(feed_dict={x: X_, y: y_}, session=sess)
        print ( "Overall Correct Predictions: ", accuracy.eval({x: X_, y: y_}),"\n")
    

    #probabilities
    if single:
        #fetch the probabilities from the output layer
        probabilities=pred
        output = probabilities.eval(feed_dict={x: X_, y: y_}, session=sess)  
        
        #Thresholding the output and convert correct predictions to 1 and others to 0
        output[output < thres] = 0
        output[output > thres] = 1
        y_ = y_.astype(float)
        
        #return indexes of correct predicted labels
        trueClasses = np.count_nonzero(output)
        indexes = np.array([np.argsort(output)])
        cat = categories[indexes]
        
        #count number of correct predictions
        for i in range(len(y_[0])):
            if(y_[0][i] == 1 and output[0][i] == 1):#TP
                TP = TP+1
                
            if(y_[0][i] == 1 and output[0][i] == 0):#FN
                FN = FN+1 
                
            if(y_[0][i] == 0 and output[0][i] == 1):#FP
                FP = FP+1   
                
            if(y_[0][i] == 0 and output[0][i] == 0):#TN
                TN = 0
                
        #Check all multi-labels and return true if matched else false
        if (output==y_).all():
            correct = True
        else:
            correct = False
         
        #Checking output of predicted labels
        expOut = []        
        misc = False
        for i in range(len((np.where(y_[:]==1))[1])):
            expOut.append(categories[(np.where(y_[:]==1))[1][i]])
            
            if (np.where(y_[:]==1))[1][i] == 9 :
                misc = True
       
        if misc == True and len(cat[0][0][12-trueClasses:12])==0:
            TP = TP+1
            FN = FN-1
            output[0][9] = 1
            
        if debug == True and correct == False:
            if sent != "":
                print ( point,"# ", sent)
            print ( "---- False Prediction ----")

        elif debug == True and correct == True:
            if sent != "":
                print ( point,"# ", sent)
            print ( "---- Correct Prediction ----")
        
        if debug == True :
            print ( "\nMulti-hot enocodings")
            print ( "True Label: ",y_)
            print ( "Pred Label: ",output)
            print ( "TP:", TP, "FN:", FN, "TN:",TN , "FP:", FP)

            print ( "\nCategory Labels")
            print ( "True Labels     : ", expOut)
            print ( "Predicted Labels: ",cat[0][0][12-trueClasses:12], "\n-----------------------------------------------------\n")
        
    return correct, output, TP, FP, TN, FN
    
    
def splitDataset(examples, labels, split=0.8):  
    data = list(zip(examples,labels))
    random.shuffle(data)
    data_length = int(round(split*len(data)))
    train = data[:data_length] #training set
    validation = data[data_length:]#validation set
    train = list(zip(*train))
    validation = list(zip(*validation))
    x_train = pd.Series(np.array(train[0]))
    y_train = np.array(train[1])
    x_val   = pd.Series(np.array(validation[0]))
    y_val   = np.array(validation[1])
    return x_train, y_train, x_val, y_val

def multiCategoryVectors(dataset, classes=12):   
    labels = np.zeros((dataset.shape[0], classes))#matrix of zeroes for accross all example labels
    for i in range(dataset.shape[0]):  
        vec = dataset.values[i][2:14]
        labels[i] = vec
    print ( "Hot encoded vectors shape: ",labels.shape)
    return labels
        
def clean(sentences, remove_stopwords=False):
    sentences = [review_to_wordlist(review,remove_stopwords) for review in sentences]
    return sentences


def cosine(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))


def fixed_Vector_Size(first, second, chunk, fromStart=True):
    if fromStart == True:
        red_first  = first[0:(first.size*chunk)]
        red_second = second[0:(first.size*chunk)]
    elif fromStart == False:
        strt = (first.size-first.size*chunk)
        red_first  = first[strt:first.size]
        red_second = second[strt:first.size]
        
    #print ( "Dimensions: ", red_first.shape, red_second.shape)
    #print ( red_first, red_second)
    return cosine(red_first, red_second)

def firstN_use(first, chunk, fromStart=True):
    if fromStart == True:
        red_first  = first[0:(first.size*chunk)]
    elif fromStart == False:
        strt = (first.size-first.size*chunk)
        red_first  = first[strt:first.size]
    return red_first

def randChunkOfN_use(first, chunk):
    n = first.size*chunk
    randRange = first.size-n
    randomChunk = random.randint(0,randRange)+1    
    red_first  = first[randomChunk:(randomChunk+n)]
    return red_first


def randChunkOfNSize(first, second, chunk):
    n = first.size*chunk
    randRange = first.size-n
    randomChunk = random.randint(0,randRange)+1
    
    red_first  = first[randomChunk:(randomChunk+n)]
    red_second = second[randomChunk:(randomChunk+n)]
    #print ( "Dimensions: ", red_first.shape, red_second.shape)
    #print ( red_first, red_second)
    return cosine(red_first, red_second)

def wordAvg(vec, chunk):
    n          = (vec.size*chunk)
    vector     = np.zeros(n).reshape((n))
    totalFolds = int(math.ceil(vec.size/n))+1
    
    count=0
    for inc in range(1,totalFolds):
        index = inc*n
        if(index>vec.size):
            index = index-(index-vec.size)
            vector[0: len(vec[count*n:index])] = vec[count*n:index]
        else:
            vector += vec[count*n:index]
        count = count+1
        
    vector /= count
    return vector
        
    
def randomOrder(arr):
    arr_ = arr.copy()
    random.shuffle(arr_)
    return arr_
    
def fixed_meanVector(vec, chunk):
    size = (vec.size*chunk)
    R    = (vec.size/size)
    pad_size = math.ceil(float(vec.size)/R)*R - vec.size
    vec_padded = np.append(vec, np.zeros(pad_size)*np.NaN)
    
    print ( "Org Vector: ",vec.size, "output Size: ",size, "Windows Size: ",R, "Padding size", pad_size)
    newVec = scipy.nanmean(vec_padded.reshape(-1,R), axis=1)
    #print ( "New Vector shape: ",newVec.shape)
    #print ( newVec)
    return newVec

def resized(data,M,N):    
    res=np.empty(N,data.dtype)
    value=0
    n=1
    m=0
    while n<=N:
        while m*N<n*M :
            value+=data[m]
            m+=1
        carry=(m-n*M/N)*data[m-1]
        value-=carry
        res[n-1]=value*N/M
        n+=1
        value=carry
    return res

def meanVec_two(vec, chunk):
    M      = vec.size
    N      = (vec.size*chunk)
    newVec = resized(vec,M,N)
    #print ( "New Vector shape: ",newVec.shape,"\n", newVec)
    return newVec

def valueToPercent(M, N):
    return (N/M*100)

def sentenceStats(listData, name):
    dataLengths = []
    avg = 0
    sequence_length = max(len(x) for x in listData)
    for x in listData:
        avg += len(x)
        dataLengths.append(len(x))
        #print ( len(x)
    avg = avg/len(listData)
    print ( name, np.std(dataLengths),"Max sentence: ",sequence_length, "Avg sentence",avg)
    
    
def stats(vector1, vector2, chunk):
    actlCosAngle   = cosine(vector1, vector2)
    firstnCosAngle = fixed_Vector_Size(vector1, vector2, chunk, fromStart=True)
    lastnCosAngle  = fixed_Vector_Size(vector1, vector2, chunk, fromStart=False)
    randCosAngle   = randChunkOfNSize(vector1,vector2, chunk)
    meanCosAngle   = cosine(meanVec_two(vector1, chunk),meanVec_two(vector2, chunk))
    return actlCosAngle, firstnCosAngle,lastnCosAngle, randCosAngle, meanCosAngle

def avgWordVector(vector1, vector2, chunk):
    return cosine(wordAvg(vector1, chunk), wordAvg(vector2, chunk))

def getWordChutext(text, maxlen):
    return maxlen/len(text)