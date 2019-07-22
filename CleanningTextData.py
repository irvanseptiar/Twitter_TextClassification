import TextMining as tm
import pandas as pd
import time
import os

def opennewfile(path):
    result = [] 
    files = os.listdir(path)
    for name in files:
        result.append(name)
    result.sort(reverse = True)

    return(path+'/'+result[0])

#def cleanningtext (data):     
#    start_time = time.time()
#    cleantweet = [] 
#    for i in range(0, len(data)):
#        print("Cleaning Tweets Indeks ke-", i)
#        tweet = tm.cleanText(data['text'][i],fix=SlangS, pattern2 = False, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, min_charLen = 1)
#        tweet = tm.
#        print(tweet)
#        cleantweet.append(tweet)
#    print("%s seconds" %(time.time()-start_time))
#    
#    Dataclean1 = pd.DataFrame(cleantweet)
#    Dataclean1.columns = ['Cleaned_Tweet']
#    return (Dataclean1)
    
def cleanningtext (data, both = True, onlyclean = False, dropduplicate = False):
    start_time = time.time()
    cleantweet = []
    
    
    for i in range(0, len(data)):
        print("Cleaning Tweets Indeks ke-", i)
        if both:
            tweet = tm.cleanText(data['Tweet'][i],fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, min_charLen = 2)
            tweet = tm.handlingnegation(tweet)
            print(tweet)
            cleantweet.append(tweet)
#            print("%s seconds" %(time.time()-start_time))
#        elif onlyclean:
#            tweet = tm.cleanText(data['Tweet'][i],fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, min_charLen = 3)
#            #tweet = tm.handingnegation(tweet)
#            print(tweet)
#            cleantweet.append(tweet)
#            print("%s seconds" %(time.time()-start_time))
        else:
            tweet = tm.handlingnegation(data['Tweet'][i])
            print(tweet)
            cleantweet.append(tweet)
    
    print("%s seconds" %(time.time()-start_time))

    if dropduplicate:
        Dataclean = pd.DataFrame(cleantweet)
        Dataclean1 = Dataclean.drop_duplicates(keep='first')
        Dataclean1.columns = ['Tweet']
        indexdata  =  Dataclean1.index.values    
        targetbaru = []
        for i in range(0,len(indexdata)):
            trgt = data.Label[indexdata[i]]
            targetbaru.append(trgt)
    
        target = pd.DataFrame(targetbaru)
        target.columns = ['Label']
    
        Dataclean2 = Dataclean1.reset_index(drop='True')
        data1 = pd.concat([Dataclean2,target],axis=1)
    else:
        Dataclean1 = pd.DataFrame(cleantweet)
        Dataclean1.columns = ['Tweet']
        data1 = pd.concat([Dataclean1,data.Label],axis=1)
        
    return (data1)

data = pd.read_excel('datafiltertext.xlsx')

fSlang = opennewfile('slangword')
bahasa = 'id'
stops, lemmatizer = tm.LoadStopWords(bahasa, sentiment = True)
sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}

databersih = cleanningtext(data)
datagab = pd.concat([data,databersih], axis = 1)
datagab.to_excel('databersihtext1.xlsx', index = False)