import pandas as pd
import sklearn
from sentence_transformers import SentenceTransformer,util
import numpy as np
n='TargetDataBasecsv.csv'
m='completedclient.csv'
df1=pd.read_csv(m)
df=pd.read_csv(n)

ym=df.columns.values.tolist()
print(ym)

ym1=df1.columns.values.tolist()
print(ym1)
model_name="stsb-roberta-large"
print(model_name)


model = SentenceTransformer(model_name)
embeddings1 = model.encode(ym, convert_to_tensor=True)
embeddings2 = model.encode(ym1, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(embeddings2, embeddings1)
#a=[]
#b=[]
#score=[]
d=[]
t=[]

for i in range(len(ym1)):
    for j in range(len(ym)):
        #a.append(ym[i])
        #b.append(ym1[i])
        #score.append(cosine_scores[i][j].item())
        #data={'word1':a,'word2':b,'similarity score':score}

        t.append(ym1[i])
        t.append(ym[j])
        t.append(cosine_scores[i][j].item())
        d.append(t)
        t=[]
#print(d)
        
        #print("word 1:", ym[i])
        #print("word 2:", ym1[j])
        #print("Similarity Score:", cosine_scores[i][j].item())
        #print()

f=pd.DataFrame(d,columns=["Source","Target","Match"])
print(f)

dicte={}
for i in range(len(f)):
    tg=[]                                
    dicte1={}
    if f.Source[i] in dicte.keys():
        if f.Match[i]>dicte[f._get_value(i,'Source')][1]:
            tg.append(f.Target[i])
            tg.append(f.Match[i])
            dicte1[f.Source[i]]=tg
            dicte.update(dicte1)
        
    else:
        tg.append(f.Target[i])
        tg.append(f.Match[i])
        dicte[f.Source[i]]=tg
        

sour=[]
for i in dicte.keys():
    sour.append(i)

tar=[]
for i in sour:
    tar.append(dicte[i][0])

per=[]
for i in sour:
    per.append(dicte[i][1])

souro=pd.DataFrame(sour,columns=['Source'])
taro=pd.DataFrame(tar,columns=['Target'])
pero=pd.DataFrame(per,columns=['Match'])
fin=pd.merge(souro,taro,right_index=True, left_index=True)
fino=pd.merge(fin,pero,right_index=True, left_index=True)
print(fino)

#drf=pd.DataFrame(data)
#print(drf)

