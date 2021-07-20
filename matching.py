import pandas as pd
import numpy as np
import tensorflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

n='TargetDataBasecsv.csv'
m='completedclient.csv'
df=pd.read_csv(n)
df1=pd.read_csv(m)

first=df1.iloc[0].tolist()

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
'''
def sample(first):
    return first
x=np.vectorize(sample,otypes=[np.ndarray])
a = np.arange(19)
print(x(a))
#print(x)
'''
newdf=[]
for i in first:
    res=isinstance(i,str)
    if res:
        newdf.append(i)
print(newdf)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newdf)
print(X.toarray())
print(X.shape)

vectorizer1 =  TfidfVectorizer()
Y = vectorizer1.fit_transform(newdf)
print(Y.toarray())
print(Y.shape)
#print(vectorizer.get_feature_names())
#print(temp)

voc_size=10000
onehot_repr=[one_hot(words,voc_size)for words in newdf] 
#print(onehot_repr)

sent_length=2
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
print(embedded_docs.shape)

# column vectors

ym=df.columns.values.tolist()
vectorizer3 = CountVectorizer()
X1 = vectorizer.fit_transform(ym)
print(X1.toarray())
print(X1.shape)

vectorizer4 =  TfidfVectorizer()
Y1 = vectorizer4.fit_transform(ym)
print(Y1.toarray())
print(Y1.shape)

voc_size1=10000
onehot_repr1=[one_hot(words,voc_size1)for words in ym] 
#print(onehot_repr1)

sent_length=2
embedded_docs1=pad_sequences(onehot_repr1,padding='pre',maxlen=sent_length)
print(embedded_docs1)
print(embedded_docs1.shape)

sim=cosine_similarity(embedded_docs,embedded_docs1)
sim1=euclidean_distances(embedded_docs,embedded_docs1)
print(sim)
print(sim1)







