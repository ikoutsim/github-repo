import os
from math import ceil, log
from nltk import word_tokenize, sent_tokenize
import re
import nltk
from nltk.util import ngrams
#from nltk.tokenize import RegexpTokenizer
from numpy import prod
from random import shuffle, choice
import csv
import pickle
from datetime import datetime

#Global variables
g1 = None
g2 = None
Vocab = None
corpus = None
N = 0

def Tokenize(l,n, CheckPrint = False):

    # Add starts & ends
    l = "".join(["*start" + str(i) +"* " for i in range(1,n)]) + l + " *end*"
    
#    #tokenize
    tokens = word_tokenize(l)

    #remove common punctuation
    tokens = [w for w in tokens if re.match('[().,-]',w) is None 
              and re.match('^[^A-Za-z]*$',w) is None 
              and re.match('^[<(].*[)>]$',w) is None]

    #replace uncommon words with UNK 
    tokens = [w if w in Vocab else '*UNK*' for w in tokens]

    return tokens

def PrepCorpus(CorpusPercentage = 0.1):
    
    path = os.path.realpath('.') + r'\txt\en'
    files = []    

    #complile list of files available
    for filename in os.listdir(path):
        files.append(path + '\\' + filename)
    #shuffle the list
    shuffle(files)
    
    #sample of the files
    files = files[:ceil(len(files)*CorpusPercentage)]
                 
    corp = []
    for f in files:
    
        with open(f, newline='', encoding='utf-8') as ff:
            
            reader = csv.reader(ff)
            
            for row in reader:
                
                if len(row)==0:
                    continue
                
                #Filter out HTML and context
                if re.match('[<(].*[)>]', row[0]) is None:
                    corp.extend(sent_tokenize("".join(row)))
                    
    return corp               

def PrepGrams(n, CorpusPercentage = 0.1, Corp = None):

    global g1, g2, Vocab, corpus
    #global line
    
    if Corp is None:            
        corpus = PrepCorpus(CorpusPercentage)
    else:
        corpus = Corp
        
    print("Preparing Vocab")                
    
    #corpus ready in a list of sentences
    #preparing list of words
    corpus_words = []
    for r in corpus:
        corpus_words.extend([w for w in r.split(' ')])
    #convert o nltk's counter
    corpus_words = nltk.FreqDist(corpus_words)    

    #prepare vocabulary, and exclude infrequent words, and frequent punctuation
    Vocab = [w for w in corpus_words if corpus_words[w] >= 10 
             and re.match('^[^A-Za-z]*$',w) is None]
    #clear memory of corpus_words
    del corpus_words
    
    #add starts and ends to the vocabulary      
    Vocab.extend(['*end*'])
    Vocab.extend(["*start" + str(i) +"*" for i in range(1,n)])
        
    #declare empty counters of ngrams and (n-1)grams        
    g1 = nltk.FreqDist([])
    g2 = nltk.FreqDist([])

    print("Preparing Ngrams")
    
    #iterate lines
    for line in corpus:
        
        #tokenize
        tks = Tokenize(line, n)
        
        #add them to global count of ngrams
        g2 += nltk.FreqDist([ gram for gram in ngrams(tks, n) ])
        g1 += nltk.FreqDist([ gram for gram in ngrams(tks, n-1) ])
    
    print("Ngrams ready")

def ProbGram(wk):
    #print(wk,g2[wk], '\t', wk[:-1], g1[wk[:-1]])
    return (g2[wk] + 1) / (g1[wk[:-1]] + len(Vocab))

    
def ProbSequence(s,n, case = 1, IncludeStarts = True):
    global N
    
    #tokenize
    tks = Tokenize(s, n)
    #convert to ngrams
    probs = [ProbGram(gram) for gram in ngrams(tks, n)]
    #exclude starts if required
    if not IncludeStarts:
        probs = probs[n:]
    
    #add to length of test corpus (used for cross entropy)
    N += len(probs)

    if case == 1:
        return prod(probs)
    elif case == 2:
        return sum([-log(y,2) for y in probs])
    elif case == 3: 
        return sum([-log(y,2)*y for y in probs])

def CompareLogProbs(n):
    
    StringOK = None
    while StringOK != 'y':
        s = choice(corpus)
        StringOK = input("Is the following sentence acceptable? [y/n]" + '\n\t' + s + '\n')
    print("The Log Prob = {0}".format(round(ProbSequence(s,n, 2),2),'\n'))

    nWords = len(s.split())
    s = ''
    for i in range(nWords):
        s += choice(Vocab) + " "
    
    print("Random Sentence:",'\n\t', s)
    print("The Log Prob = {0}".format(round(ProbSequence(s,n, 2),2),'\n'))

def StoreObj(zObj,zName):
    f = open(zName + ".pkl",'wb')
    pickle.dump(zObj, f)
    f.close()
    
    
def LoadObj(zName):
    f = open(zName + '.pkl', 'rb')
    cfd = pickle.load(f)
    f.close()
    return cfd
    
def WordPredictor(zSentence, n):
    #preced is a tuple, of: 
        #size 2 in trigram models    
        #size 1 in bigram model
    
    zw = None #thats the word we are predicting
    zc = 0 #thats the counter of the ngram containing preced and zw
    
    #will tokenize, and retrieve only the relevant ones
    tks = word_tokenize(zSentence)[-n+1:]
    tks = [w if w in Vocab else '*UNK*' for w in tks]

    for w in g2:        
        match = False
        for i in range(len(tks)):
            if w[i] == tks[i]:
                match = True
            else: 
                match = False
                break
        
        if match:
            if g2[w] > zc and w != "*end*" and w != "*UNK*":
                zw = w[-1]
                zc = g2[w]

    #applying backoff
    if zw is None and n > 2:
        for w in g1:        
            match = False
            for i in range(len(tks)):
                if w[i] == tks[i]:
                    match = True
                else: 
                    match = False
                    break
            
            if match:
                if g1[w] > zc and w != "*end*" and w != "*UNK*":
                    zw = w[-1]
                    zc = g1[w]

    if zw is None:
        #revert to unigram model
        #select most common word from vocab that isnt *UNK* *end* or *start*
        zw = Vocab.most_common(0)
        
     
    return zw

def CrossEntropy(n, testcorp):

    #assign the starts and ends
    global N
    
    N = 0
    cEntropy = 0

    for i in range(len(testcorp)):
        testcorp[i] = "".join(["*start" + str(i) +"* " for i in range(1,n)]) + testcorp[i] + " *end*"
        cEntropy += ProbSequence(testcorp[i],n, 1, False)      

    return cEntropy / N
                
def Load(n):
    global corpus, Vocab, g1, g2

    g1 = LoadObj('g1n' + str(n))
    g2 = LoadObj('g2n' + str(n))
    
    Vocab = LoadObj('Vocab')
    corpus = LoadObj('corpus')
    
def Save(n):
    global corpus, Vocab, g1, g2
    
    StoreObj(g1,'g1n' + str(n))
    StoreObj(g2,'g2n' + str(n))

    StoreObj(Vocab,'Vocab')
    StoreObj(corpus,'corpus')

def TrainModels(PctFiles):
    global corpus, Vocab, g1, g2
    model = 2
    t = datetime.now()   
    PrepGrams(model,PctFiles)
    Save(model)
    print("Time Ngramming:",'\t',datetime.now() - t, '\tn = ',model)
    
    model = 3
    t = datetime.now()   
    PrepGrams(model,Corp = corpus)
    Save(model)
    print("Time Ngramming:",'\t',datetime.now() - t, '\tn = ',model)
    
def RunMain():
    
    global corpus, Vocab, g1, g2
    
    ToLoad = False
    model = 2
    
    if ToLoad:
        Load(model)
    else:
        TrainModels(0.05)
        
    
    s = input("Please Enter Sentence to Calculate its Probability: \n")
    print("Probability = {0}".format(ProbSequence(s,model)),'\n')
#    
    CompareLogProbs(model)

    s = input("Please Enter Sentence to be Completed by Model: \n")
    print (WordPredictor("What should the delagates",model))
    
    testcorp = PrepCorpus(0.005)
    Load(2)
    x = CrossEntropy(2, testcorp)
    print("Cross Entropy of Bigram Model:",x)
    print("Perplexity of Bigram Model:",2**x)
    Load(3)
    x = CrossEntropy(3, testcorp)
    print("Cross Entropy of Trigram Model:", x)
    print("Perplexity of Trigram Model:", 2**x)
        

RunMain()
    

