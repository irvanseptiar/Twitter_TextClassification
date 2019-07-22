import numpy as np 
from keras.preprocessing.sequence import pad_sequences
import TextMining as tm
from keras.models import load_model
import tensorflow as tf
import pickle
from opennewfile import opennewfile

def loadtokenizer(filepath):
    with open(filepath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return (tokenizer)

def loadmodel(filepath):
    global model
    model = load_model(filepath)
    global graph
    graph = tf.get_default_graph()
    return(model)

def preprocess(text, lemmit = True):
    if lemmit:
        text = tm.cleanText(text,fix=SlangS, pattern2 = False, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, min_charLen = 2)
        text = tm.handlingnegation(text)
    else:
        text = tm.cleanText(text,fix = SlangS, lang = bahasa, stops = stops, lemma= None,symbols_remove=True,min_charLen=3)
        text = tm.handlingnegation(text)
    return(text)

def maxposition(array):
   array = list(array) 
   maxposition = array.index(max(array)) 
   return maxposition

def predfiltertext(text, lemma = True):
    text = str(text)
    text = [preprocess(text, lemmit = lemma)]
    text = tokenizersen.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=150 ,dtype = 'int32', value = 0)
    with graph.as_default():
        filtertext = model.predict(text,batch_size=1,verbose = 2)[0]
    labels = ['Porn Content', 'Advertaisment', 'Others']
    result = labels[np.argmax(filtertext)]
#    print(pred, labels[np.argmax(pred)])
#    score = maxposition(filtertext)
#    if score == 0:
#        result = 'Porn Content'
#    elif score == 1:
#        result = 'Advertaisment'
#    else:
#        result = 'Others'
    return filtertext, result

fSlang = opennewfile(path = 'slangword')
bahasa = 'id'
stops, lemmatizer = tm.LoadStopWords(bahasa, sentiment = True)
sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}
tokenizersen = loadtokenizer(opennewfile(path = 'tokenizer'))
model = loadmodel(opennewfile(path = 'model'))  



