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
    model_name="stsb-roberta-large"
    print(model_name)


    model = SentenceTransformer(model_name)
    embeddings1 = model.encode(ym, convert_to_tensor=True)
    embeddings2 = model.encode(ym1, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
#a=[]
#b=[]
#score=[]
    d=[]
    t=[]

    for i in range(len(ym)):
        for j in range(len(ym1)):
        #a.append(ym[i])
        #b.append(ym1[i])
        #score.append(cosine_scores[i][j].item())
        #data={'word1':a,'word2':b,'similarity score':score}



            t.append(ym1[j])
            t.append(ym[i])
            t.append(cosine_scores[i][j].item())
            d.append(t)
            t=[]
        
        #print("word 1:", ym[i])
        #print("word 2:", ym1[j])
        #print("Similarity Score:", cosine_scores[i][j].item())
        #print()

    f=pd.DataFrame(d,columns=["Source","Target","Match"])

    dicte={}
    for i in range(len(f)):
        tg=[]                                
        dicte1={}
        if f.Target[i] in dicte.keys():
            if f.Match[i]>dicte[f._get_value(i,'Target')][1]:
                tg.append(f.Source[i])
                tg.append(f.Match[i])
                dicte1[f.Target[i]]=tg
                dicte.update(dicte1)
        
        else:
            tg.append(f.Source[i])
            tg.append(f.Match[i])
            dicte[f.Target[i]]=tg
        

    tar=[]
    for i in dicte.keys():
        tar.append(i)

    sour=[]
    for i in tar:
        sour.append(dicte[i][0])

    per=[]
    for i in tar:
        per.append(dicte[i][1])

    souro=pd.DataFrame(sour,columns=['Source'])
    taro=pd.DataFrame(tar,columns=['Target'])
    pero=pd.DataFrame(per,columns=['Match'])
    fin=pd.merge(souro,taro,right_index=True, left_index=True)
    fino=pd.merge(fin,pero,right_index=True, left_index=True)
    #st.dataframe(fino)
    return fino,ym1

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
            st.dataframe(df)
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
        col1.text(df.Target[0])
        col2.selectbox("source",[df.Source[0],client],key=1)
        col3.text(df.Match[0])

        col4,col5,col39=st.beta_columns(3)
        col4.text(df.Target[1])
        col5.selectbox("source",[df.Source[1],client],key=2)
        col39.text(df.Match[1])

        col6,col7,col8=st.beta_columns(3)
        col6.text(df.Target[2])
        col7.selectbox("source",[df.Source[2],client],key=3)
        col8.text(df.Match[2])
       
        col9,col10,col11=st.beta_columns(3)
        col9.text(df.Target[3])
        col10.selectbox("source",[df.Source[3],client],key=5)
        col11.text(df.Match[3])

        col12,col13,col14=st.beta_columns(3)
        col12.text(df.Target[4])
        col13.selectbox("source",[df.Source[4],client],key=6)
        col14.text(df.Match[4])

        col15,col16,col17=st.beta_columns(3)
        col15.text(df.Target[5])
        col16.selectbox("source",[df.Source[5],client],key=7)
        col17.text(df.Match[5])

        col18,col19,col20=st.beta_columns(3)
        col18.text(df.Target[6])
        col19.selectbox("source",[df.Source[6],client],key=8)
        col20.text(df.Match[6])

        col21,col22,col23=st.beta_columns(3)
        col21.text(df.Target[7])
        col22.selectbox("source",[df.Source[7],client],key=9)
        col23.text(df.Match[7])

        col24,col25,col26=st.beta_columns(3)
        col24.text(df.Target[8])
        col25.selectbox("source",[df.Source[8],client],key=10)
        col26.text(df.Match[8])

        col27,col28,col29=st.beta_columns(3)
        col27.text(df.Target[9])
        col28.selectbox("source",[df.Source[9],client],key=11)
        col29.text(df.Match[9])

        col30,col31,col32=st.beta_columns(3)
        col30.text(df.Target[10])
        col31.selectbox("source",[df.Source[10],client],key=12)
        col32.text(df.Match[10])

        col33,col34,col35=st.beta_columns(3)
        col33.text(df.Target[11])
        col34.selectbox("source",[df.Source[11],client],key=13)
        col35.text(df.Match[11])

        col36,col37,col38=st.beta_columns(3)
        col36.text(df.Target[12])
        col37.selectbox("source",[df.Source[12],client],key=4)
        col38.text(df.Match[12])








    elif selectbox==Menu[4]:
        pass


if __name__=="__main__":
    main()





