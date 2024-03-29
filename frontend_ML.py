import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sentence_transformers import SentenceTransformer,util

#global model1

def bert(model_name):
        n='TargetDataBasecsv.csv'
        m='completedclient.csv'
    
        df=pd.read_csv(n)
        df1=pd.read_csv(m)

        ym=df.columns.values.tolist()
    #print(ym)

        ym1=df1.columns.values.tolist()
#print(ym1)
        #model_name='nli-roberta-base'
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

        td=[]
        te=[]
        fino['Data']=''
        for j in range(len(fino['Source'])):
            for i in df1[fino['Source'][j]]:
                td.append(i)
            fino.at[j, 'Data'] =td
            td=[]
            
    #st.dataframe(fino)
    
        
        df=fino
        col1,col2,col3,col58=st.beta_columns(4)
        c1=col1.text(df.Source[0])
        t1=col2.selectbox("Target",[df.Target[0],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=1)
        m1=col3.text(df.Match[0])
        d1=col58.selectbox("Values",df.Data[0],key=58)


        col4,col5,col39,col59=st.beta_columns(4)
        c2=col4.text(df.Source[1])
        t2=col5.selectbox("Target",[df.Target[1],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=2)
        m2=col39.text(df.Match[1])
        d2=col59.selectbox("Values",df.Data[1],key=59)
    
        col6,col7,col8,col60=st.beta_columns(4)
        c3=col6.text(df.Source[2])
        t3=col7.selectbox("Target",[df.Target[2],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=3)
        m3=col8.text(df.Match[2])
        d3=col60.selectbox("Values",df.Data[2],key=60)
       
        col9,col10,col11,col61=st.beta_columns(4)
        c4=col9.text(df.Source[3])
        t4=col10.selectbox("Target",[df.Target[3],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=5)
        m4=col11.text(df.Match[3])
        d4=col61.selectbox("Values",df.Data[3],key=61)


        col12,col13,col14,col62=st.beta_columns(4)
        c5=col12.text(df.Source[4])
        t5=col13.selectbox("Target",[df.Target[4],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=6)
        m5=col14.text(df.Match[4])
        d5=col62.selectbox("Values",df.Data[4],key=62)

        col15,col16,col17,col63=st.beta_columns(4)
        c6=col15.text(df.Source[5])
        t6=col16.selectbox("Target",[df.Target[5],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=7)
        m6=col17.text(df.Match[5])
        d6=col63.selectbox("Values",df.Data[5],key=63)

        col18,col19,col20,col64=st.beta_columns(4)
        c7=col18.text(df.Source[6])
        t7=col19.selectbox("Target",[df.Target[6],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=8)
        m7=col20.text(df.Match[6])
        d7=col64.selectbox("Values",df.Data[6],key=64)


        col21,col22,col23,col65=st.beta_columns(4)
        c8=col21.text(df.Source[7])
        t8=col22.selectbox("Target",[df.Target[7],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=9)
        m8=col23.text(df.Match[7])
        d8=col65.selectbox("Values",df.Data[7],key=65)

        col24,col25,col26,col66=st.beta_columns(4)
        c9=col24.text(df.Source[8])
        t9=col25.selectbox("Target",[df.Target[8],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=10)
        m9=col26.text(df.Match[8])
        d9=col66.selectbox("Values",df.Data[8],key=66)

        col27,col28,col29,col67=st.beta_columns(4)
        c10=col27.text(df.Source[9])
        t10=col28.selectbox("Target",[df.Target[9],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=11)
        m10=col29.text(df.Match[9])
        d10=col67.selectbox("Values",df.Data[9],key=67)

        col30,col31,col32,col68=st.beta_columns(4)
        c11=col30.text(df.Source[10])
        t11=col31.selectbox("Target",[df.Target[10],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=12)
        m11=col32.text(df.Match[10])
        d11=col68.selectbox("Values",df.Data[10],key=68)

        col33,col34,col35,col69=st.beta_columns(4)
        c12=col33.text(df.Source[11])
        t12=col34.selectbox("Target",[df.Target[11],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=13)
        m12=col35.text(df.Match[11])
        d12=col69.selectbox("Values",df.Data[11],key=69)

        col36,col37,col38,col70=st.beta_columns(4)
        c13=col36.text(df.Source[12])
        t13=col37.selectbox("Target",[df.Target[12],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=4)
        m13=col38.text(df.Match[12])
        d13=col70.selectbox("Values",df.Data[12],key=70)

        
        col40,col41,col42,col71=st.beta_columns(4)
        c14=col40.text(df.Source[13])
        t14=col41.selectbox("Target",[df.Target[13],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=14)
        m14=col42.text(df.Match[13])
        d14=col71.selectbox("Values",df.Data[13],key=71)

        col43,col44,col45,col72=st.beta_columns(4)
        c15=col43.text(df.Source[14])
        t15=col44.selectbox("Target",[df.Target[14],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=15)
        m15=col45.text(df.Match[14])
        d15=col72.selectbox("Values",df.Data[14],key=72)

        
        col46,col47,col48,col73=st.beta_columns(4)
        c16=col46.text(df.Source[15])
        t16=col47.selectbox("Target",[df.Target[15],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=16)
        m16=col48.text(df.Match[15])
        d16=col73.selectbox("Values",df.Data[15],key=73)
        
        col49,col50,col51,col74=st.beta_columns(4)
        c17=col49.text(df.Source[16])
        t17=col50.selectbox("Target",[df.Target[16],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=17)
        m17=col51.text(df.Match[16])
        d17=col74.selectbox("Values",df.Data[16],key=74)

        col52,col53,col54,col75=st.beta_columns(4)
        c18=col52.text(df.Source[17])
        t18=col53.selectbox("Target",[df.Target[17],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=18)
        m18=col54.text(df.Match[17])
        d18=col75.selectbox("Values",df.Data[17],key=75)

        col55,col56,col57,col76=st.beta_columns(4)
        c19=col55.text(df.Source[18])
        t19=col56.selectbox("Target",[df.Target[18],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=19)
        m19=col57.text(df.Match[18])
        d19=col76.selectbox("Values",df.Data[18],key=76)

        column=['Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender']
        final=pd.DataFrame(columns=column)
        #st.dataframe(final)
        if t1=='Acct_id':
            final['Acct_id']=df.Data[0]
        elif t1=='Acct_UIDNo.':
            final['Acct_UIDNo']=df.Data[0]
        elif t1=='Acct_UIDNo.':
            final['Acct_UIDNo']=df.Data[0]
        elif t1=='Acct_UIDNo.':
            final['Acct_UIDNo']=df.Data[0]
        
        
        
        st.dataframe(final)
        #st.write(t1)
        



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

            bert(opt)


        elif opt=='stsb-roberta-base':
            
            st.write("This is other bert model with little different features")
            bert(opt)

        else:
            st.write("Select one of the models to test the dataset against the present dataset")

    
    

    #elif selectbox==Menu[3]:
        #st.header("Mapped Results")
        #df=pd.DataFrame()
        #df,df1=bert()

        
    elif selectbox==Menu[4]:
        pass
        
    



if __name__=="__main__":
    main()





