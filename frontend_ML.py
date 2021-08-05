import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sentence_transformers import SentenceTransformer,util

def bert():
    n='TargetDataBasecsv.csv'
    m='completedclient.csv'
    
    df=pd.read_csv(n)
    df1=pd.read_csv(m)

    ym=df.columns.values.tolist()
#print(ym)

    ym1=df1.columns.values.tolist()
#print(ym1)
    model_name="nli-roberta-base"
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
    #print(f)
    d=[]
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
    #print(fino)



            
    #st.dataframe(fino)
    return fino,ym

def main():
    Menu=["Home","Dataset Upload","Model selector","Results","Review & Download"]
    selectbox = st.sidebar.selectbox("Modules",Menu)
    if selectbox==Menu[0]:
        st.header("welcome")
        st.write('Semantic similarity is a metric defined over a set of documents or terms,' 
    'where the idea of distance between items is based on the likeness of their meaning '
    'or semantic content as opposed to lexicographical similarity.'
    'Schema matching is the technique of identifying objects which are semantically related.'
    'In other words, schema matching is a method of finding the correspondences between the concepts'
    'of different distributed, heterogeneous data sources. Schema matching is considered one of the basic'
    'operations for schema integration and data processing.'
    'Here we find similar dataset columns compared to each other with the option of selecting between two'
    'Models for convinience.')
    elif selectbox==Menu[1]:
        st.header("Upload Data File")
        fileuploade=st.file_uploader("Upload CSV",type=["csv"])
        if fileuploade is not None:
        
            file_details={"File Name":fileuploade.name,
        "File Type":fileuploade.type,"File Size":fileuploade.size}
            st.write(file_details)
            df=pd.read_csv(fileuploade)
            #bert(df)
            print(fileuploade.name)
            source,Target=st.beta_columns(2)
            source.header("Source")
            Target.header("Target")
            col1,col2=st.beta_columns(2)
            source=df.columns.values.tolist()
            
            for p in source:
                col1.checkbox(p)
            n=pd.read_csv('TargetDataBasecsv.csv')
            target=n.columns.values.tolist()
            for p in target:
                col2.write(p)
    elif selectbox==Menu[2]:
        st.header("Model Selector")
        opt = st.radio(
    "Models",
    ('Select one','nli-roberta-base', 'stsb-roberta-base'))

        if opt == 'nli-roberta-base':
            st.write('You can use this framework to compute sentence / text embeddings for more than 100 languages.'
        ' These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning.'
         'This can be useful for semantic textual similar, semantic search, or paraphrase mining.')
        elif opt=='stsb-roberta-base':
            st.write("This is other")
        else:
            st.write("Select one of the models to test the dataset against the present dataset")
    elif selectbox==Menu[3]:
        st.header("Mapped Results")
        df=pd.DataFrame()
        client=[]
        df,client=bert()
        
        
        col1,col2,col3=st.beta_columns(3)
        col1.text(df.Source[0])
        col2.selectbox("Target",[df.Target[0],client],key=1)
        col3.text(df.Match[0])

        col4,col5,col39=st.beta_columns(3)
        col4.text(df.Source[1])
        col5.selectbox("Target",[df.Target[1],client],key=2)
        col39.text(df.Match[1])

        col6,col7,col8=st.beta_columns(3)
        col6.text(df.Source[2])
        col7.selectbox("Target",[df.Target[2],client],key=3)
        col8.text(df.Match[2])
       
        col9,col10,col11=st.beta_columns(3)
        col9.text(df.Source[3])
        col10.selectbox("Target",[df.Target[3],client],key=5)
        col11.text(df.Match[3])

        col12,col13,col14=st.beta_columns(3)
        col12.text(df.Source[4])
        col13.selectbox("Target",[df.Target[4],client],key=6)
        col14.text(df.Match[4])

        col15,col16,col17=st.beta_columns(3)
        col15.text(df.Source[5])
        col16.selectbox("Target",[df.Target[5],client],key=7)
        col17.text(df.Match[5])

        col18,col19,col20=st.beta_columns(3)
        col18.text(df.Source[6])
        col19.selectbox("Target",[df.Target[6],client],key=8)
        col20.text(df.Match[6])

        col21,col22,col23=st.beta_columns(3)
        col21.text(df.Source[7])
        col22.selectbox("Target",[df.Target[7],client],key=9)
        col23.text(df.Match[7])

        col24,col25,col26=st.beta_columns(3)
        col24.text(df.Source[8])
        col25.selectbox("Target",[df.Target[8],client],key=10)
        col26.text(df.Match[8])

        col27,col28,col29=st.beta_columns(3)
        col27.text(df.Source[9])
        col28.selectbox("Target",[df.Target[9],client],key=11)
        col29.text(df.Match[9])

        col30,col31,col32=st.beta_columns(3)
        col30.text(df.Source[10])
        col31.selectbox("Target",[df.Target[10],client],key=12)
        col32.text(df.Match[10])

        col33,col34,col35=st.beta_columns(3)
        col33.text(df.Source[11])
        col34.selectbox("Target",[df.Target[11],client],key=13)
        col35.text(df.Match[11])

        col36,col37,col38=st.beta_columns(3)
        col36.text(df.Source[12])
        col37.selectbox("Target",[df.Target[12],client],key=4)
        col38.text(df.Match[12])

        
        col40,col41,col42=st.beta_columns(3)
        col40.text(df.Source[13])
        col41.selectbox("Target",[df.Target[13],client],key=14)
        col42.text(df.Match[13])

        col43,col44,col45=st.beta_columns(3)
        col43.text(df.Source[14])
        col44.selectbox("Target",[df.Target[14],client],key=15)
        col45.text(df.Match[14])

        
        col46,col47,col48=st.beta_columns(3)
        col46.text(df.Source[15])
        col47.selectbox("Target",[df.Target[15],client],key=16)
        col48.text(df.Match[15])
        
        col49,col50,col51=st.beta_columns(3)
        col49.text(df.Source[16])
        col50.selectbox("Target",[df.Target[16],client],key=17)
        col51.text(df.Match[16])

        col52,col53,col54=st.beta_columns(3)
        col52.text(df.Source[17])
        col53.selectbox("Target",[df.Target[17],client],key=18)
        col54.text(df.Match[17])

        col55,col56,col57=st.beta_columns(3)
        col55.text(df.Source[18])
        col56.selectbox("Target",[df.Target[18],client],key=19)
        col57.text(df.Match[18])










    elif selectbox==Menu[4]:
        pass


if __name__=="__main__":
    main()





