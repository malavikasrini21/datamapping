import streamlit as st
import numpy as np
import pandas as pd
import ast
import io
from io import StringIO
from pages import utils,upload
from sentence_transformers import SentenceTransformer,util
def bert(model_name):
        n='TargetDataBasecsv.csv'
        df=pd.read_csv(n)
        df1=pd.read_csv('data.csv')
        ym=df.columns.values.tolist()
    #print(ym)
        ym1=df1.columns.values.tolist()
        st.write(len(ym1))

#print(ym1)
       
        model = SentenceTransformer(model_name)
        embeddings1 = model.encode(ym, convert_to_tensor=True)
        embeddings2 = model.encode(ym1, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embeddings2, embeddings1)
        #st.write(cosine_scores)
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
        try:
            sources,Targets,Match_Scores,Datas=st.beta_columns(4)
            sources.header("Source")
            Targets.header("Target")
            Match_Scores.header("Match Score")
            Datas.header("Data")
    
        
            df=fino
            st.dataframe(df)
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
            t14=col41.selectbox("Target",[df.Target[13],'Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender'],key=4)
            m14=col42.text(df.Match[13])
            d14=col70.selectbox("Values",df.Data[12],key=71)


        
        
        except:
            column=['Acct_id', 'Acct_UIDNo.', 'Acct_FName', 'Acct_MName', 'Acct_LName', 'Acct_Addr1', 'Acct_Addr2', 'Acct_City', 'Acct_State', 'Acct_phone', 'Acct_email', 'Acct_DOB', 'Acct_Gender']
            final=pd.DataFrame(columns=column)
            #st.dataframe(final)
            if len(ym1)==1:
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

               # st.dataframe(final)
            
            if len(ym1)==2:

                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                #st.dataframe(final)
            
            if len(ym1)==3:

                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]

                #st.dataframe(final)

            if len(ym1)==4:

                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                #st.dataframe(final)

            if len(ym1)==5:

                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                #st.dataframe(final)

            if len(ym1)==6:
            
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5]
                
                #st.dataframe(final)

            if len(ym1)==7:
            
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5] 
                
                if t7=='Acct_id':
                    final['Acct_id']=df.Data[6]
                elif t7=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[6]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[6]
                elif t7=='Acct_MName':
                    final['Acct_MName']=df.Data[6]
                elif t7=='Acct_LName':
                    final['Acct_LName']=df.Data[6]
                elif t7=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[6]
                elif t7=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[6]
                elif t7=='Acct_City':
                    final['Acct_City']=df.Data[6]
                elif t7=='Acct_State':
                    final['Acct_State']=df.Data[6]
                elif t7=='Acct_phone':
                    final['Acct_phone']=df.Data[6]
                elif t7=='Acct_email':
                    final['Acct_email']=df.Data[6]
                elif t7=='Acct_DOB':
                    final['Acct_DOB']=df.Data[6]
                elif t7=='Acct_Gender':
                    final['Acct_Gender']=df.Data[6]
                
               # st.dataframe(final)

            if len(ym1)==8:

                
            
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5] 
                
                if t7=='Acct_id':
                    final['Acct_id']=df.Data[6]
                elif t7=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[6]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[6]
                elif t7=='Acct_MName':
                    final['Acct_MName']=df.Data[6]
                elif t7=='Acct_LName':
                    final['Acct_LName']=df.Data[6]
                elif t7=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[6]
                elif t7=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[6]
                elif t7=='Acct_City':
                    final['Acct_City']=df.Data[6]
                elif t7=='Acct_State':
                    final['Acct_State']=df.Data[6]
                elif t7=='Acct_phone':
                    final['Acct_phone']=df.Data[6]
                elif t7=='Acct_email':
                    final['Acct_email']=df.Data[6]
                elif t7=='Acct_DOB':
                    final['Acct_DOB']=df.Data[6]
                elif t7=='Acct_Gender':
                    final['Acct_Gender']=df.Data[6]

                if t8=='Acct_id':
                    final['Acct_id']=df.Data[7]
                elif t8=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[7]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[7]
                elif t8=='Acct_MName':
                    final['Acct_MName']=df.Data[7]
                elif t8=='Acct_LName':
                    final['Acct_LName']=df.Data[7]
                elif t8=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[7]
                elif t8=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[7]
                elif t8=='Acct_City':
                    final['Acct_City']=df.Data[7]
                elif t8=='Acct_State':
                    final['Acct_State']=df.Data[7]
                elif t8=='Acct_phone':
                    final['Acct_phone']=df.Data[7]
                elif t8=='Acct_email':
                    final['Acct_email']=df.Data[7]
                elif t8=='Acct_DOB':
                    final['Acct_DOB']=df.Data[7]
                elif t8=='Acct_Gender':
                    final['Acct_Gender']=df.Data[7]

                #st.dataframe(final)
            
            if len(ym1)==9:
                    
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5] 
                
                if t7=='Acct_id':
                    final['Acct_id']=df.Data[6]
                elif t7=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[6]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[6]
                elif t7=='Acct_MName':
                    final['Acct_MName']=df.Data[6]
                elif t7=='Acct_LName':
                    final['Acct_LName']=df.Data[6]
                elif t7=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[6]
                elif t7=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[6]
                elif t7=='Acct_City':
                    final['Acct_City']=df.Data[6]
                elif t7=='Acct_State':
                    final['Acct_State']=df.Data[6]
                elif t7=='Acct_phone':
                    final['Acct_phone']=df.Data[6]
                elif t7=='Acct_email':
                    final['Acct_email']=df.Data[6]
                elif t7=='Acct_DOB':
                    final['Acct_DOB']=df.Data[6]
                elif t7=='Acct_Gender':
                    final['Acct_Gender']=df.Data[6]

                if t8=='Acct_id':
                    final['Acct_id']=df.Data[7]
                elif t8=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[7]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[7]
                elif t8=='Acct_MName':
                    final['Acct_MName']=df.Data[7]
                elif t8=='Acct_LName':
                    final['Acct_LName']=df.Data[7]
                elif t8=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[7]
                elif t8=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[7]
                elif t8=='Acct_City':
                    final['Acct_City']=df.Data[7]
                elif t8=='Acct_State':
                    final['Acct_State']=df.Data[7]
                elif t8=='Acct_phone':
                    final['Acct_phone']=df.Data[7]
                elif t8=='Acct_email':
                    final['Acct_email']=df.Data[7]
                elif t8=='Acct_DOB':
                    final['Acct_DOB']=df.Data[7]
                elif t8=='Acct_Gender':
                    final['Acct_Gender']=df.Data[7]
                
                if t9=='Acct_id':
                    final['Acct_id']=df.Data[8]
                elif t9=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[8]
                elif t9=='Acct_FName':
                    final['Acct_FName']=df.Data[8]
                elif t9=='Acct_MName':
                    final['Acct_MName']=df.Data[8]
                elif t9=='Acct_LName':
                    final['Acct_LName']=df.Data[8]
                elif t9=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[8]
                elif t9=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[8]
                elif t9=='Acct_City':
                    final['Acct_City']=df.Data[8]
                elif t9=='Acct_State':
                    final['Acct_State']=df.Data[8]
                elif t9=='Acct_phone':
                    final['Acct_phone']=df.Data[8]
                elif t9=='Acct_email':
                    final['Acct_email']=df.Data[8]
                elif t9=='Acct_DOB':
                    final['Acct_DOB']=df.Data[8]
                elif t9=='Acct_Gender':
                    final['Acct_Gender']=df.Data[8]

                #st.dataframe(final)

            if len(ym1)==10:
                    
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5] 
                
                if t7=='Acct_id':
                    final['Acct_id']=df.Data[6]
                elif t7=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[6]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[6]
                elif t7=='Acct_MName':
                    final['Acct_MName']=df.Data[6]
                elif t7=='Acct_LName':
                    final['Acct_LName']=df.Data[6]
                elif t7=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[6]
                elif t7=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[6]
                elif t7=='Acct_City':
                    final['Acct_City']=df.Data[6]
                elif t7=='Acct_State':
                    final['Acct_State']=df.Data[6]
                elif t7=='Acct_phone':
                    final['Acct_phone']=df.Data[6]
                elif t7=='Acct_email':
                    final['Acct_email']=df.Data[6]
                elif t7=='Acct_DOB':
                    final['Acct_DOB']=df.Data[6]
                elif t7=='Acct_Gender':
                    final['Acct_Gender']=df.Data[6]

                if t8=='Acct_id':
                    final['Acct_id']=df.Data[7]
                elif t8=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[7]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[7]
                elif t8=='Acct_MName':
                    final['Acct_MName']=df.Data[7]
                elif t8=='Acct_LName':
                    final['Acct_LName']=df.Data[7]
                elif t8=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[7]
                elif t8=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[7]
                elif t8=='Acct_City':
                    final['Acct_City']=df.Data[7]
                elif t8=='Acct_State':
                    final['Acct_State']=df.Data[7]
                elif t8=='Acct_phone':
                    final['Acct_phone']=df.Data[7]
                elif t8=='Acct_email':
                    final['Acct_email']=df.Data[7]
                elif t8=='Acct_DOB':
                    final['Acct_DOB']=df.Data[7]
                elif t8=='Acct_Gender':
                    final['Acct_Gender']=df.Data[7]
                
                if t9=='Acct_id':
                    final['Acct_id']=df.Data[8]
                elif t9=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[8]
                elif t9=='Acct_FName':
                    final['Acct_FName']=df.Data[8]
                elif t9=='Acct_MName':
                    final['Acct_MName']=df.Data[8]
                elif t9=='Acct_LName':
                    final['Acct_LName']=df.Data[8]
                elif t9=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[8]
                elif t9=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[8]
                elif t9=='Acct_City':
                    final['Acct_City']=df.Data[8]
                elif t9=='Acct_State':
                    final['Acct_State']=df.Data[8]
                elif t9=='Acct_phone':
                    final['Acct_phone']=df.Data[8]
                elif t9=='Acct_email':
                    final['Acct_email']=df.Data[8]
                elif t9=='Acct_DOB':
                    final['Acct_DOB']=df.Data[8]
                elif t9=='Acct_Gender':
                    final['Acct_Gender']=df.Data[8]
                
                
                if t10=='Acct_id':
                    final['Acct_id']=df.Data[9]
                elif t10=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[9]
                elif t10=='Acct_FName':
                    final['Acct_FName']=df.Data[9]
                elif t10=='Acct_MName':
                    final['Acct_MName']=df.Data[9]
                elif t10=='Acct_LName':
                    final['Acct_LName']=df.Data[9]
                elif t10=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[9]
                elif t10=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[9]
                elif t10=='Acct_City':
                    final['Acct_City']=df.Data[9]
                elif t10=='Acct_State':
                    final['Acct_State']=df.Data[9]
                elif t10=='Acct_phone':
                    final['Acct_phone']=df.Data[9]
                elif t10=='Acct_email':
                    final['Acct_email']=df.Data[9]
                elif t10=='Acct_DOB':
                    final['Acct_DOB']=df.Data[9]
                elif t10=='Acct_Gender':
                    final['Acct_Gender']=df.Data[9]

                #st.dataframe(final)

                
            if len(ym1)==11:
                    
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5] 
                
                if t7=='Acct_id':
                    final['Acct_id']=df.Data[6]
                elif t7=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[6]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[6]
                elif t7=='Acct_MName':
                    final['Acct_MName']=df.Data[6]
                elif t7=='Acct_LName':
                    final['Acct_LName']=df.Data[6]
                elif t7=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[6]
                elif t7=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[6]
                elif t7=='Acct_City':
                    final['Acct_City']=df.Data[6]
                elif t7=='Acct_State':
                    final['Acct_State']=df.Data[6]
                elif t7=='Acct_phone':
                    final['Acct_phone']=df.Data[6]
                elif t7=='Acct_email':
                    final['Acct_email']=df.Data[6]
                elif t7=='Acct_DOB':
                    final['Acct_DOB']=df.Data[6]
                elif t7=='Acct_Gender':
                    final['Acct_Gender']=df.Data[6]

                if t8=='Acct_id':
                    final['Acct_id']=df.Data[7]
                elif t8=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[7]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[7]
                elif t8=='Acct_MName':
                    final['Acct_MName']=df.Data[7]
                elif t8=='Acct_LName':
                    final['Acct_LName']=df.Data[7]
                elif t8=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[7]
                elif t8=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[7]
                elif t8=='Acct_City':
                    final['Acct_City']=df.Data[7]
                elif t8=='Acct_State':
                    final['Acct_State']=df.Data[7]
                elif t8=='Acct_phone':
                    final['Acct_phone']=df.Data[7]
                elif t8=='Acct_email':
                    final['Acct_email']=df.Data[7]
                elif t8=='Acct_DOB':
                    final['Acct_DOB']=df.Data[7]
                elif t8=='Acct_Gender':
                    final['Acct_Gender']=df.Data[7]
                
                if t9=='Acct_id':
                    final['Acct_id']=df.Data[8]
                elif t9=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[8]
                elif t9=='Acct_FName':
                    final['Acct_FName']=df.Data[8]
                elif t9=='Acct_MName':
                    final['Acct_MName']=df.Data[8]
                elif t9=='Acct_LName':
                    final['Acct_LName']=df.Data[8]
                elif t9=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[8]
                elif t9=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[8]
                elif t9=='Acct_City':
                    final['Acct_City']=df.Data[8]
                elif t9=='Acct_State':
                    final['Acct_State']=df.Data[8]
                elif t9=='Acct_phone':
                    final['Acct_phone']=df.Data[8]
                elif t9=='Acct_email':
                    final['Acct_email']=df.Data[8]
                elif t9=='Acct_DOB':
                    final['Acct_DOB']=df.Data[8]
                elif t9=='Acct_Gender':
                    final['Acct_Gender']=df.Data[8]
                
                
                if t10=='Acct_id':
                    final['Acct_id']=df.Data[9]
                elif t10=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[9]
                elif t10=='Acct_FName':
                    final['Acct_FName']=df.Data[9]
                elif t10=='Acct_MName':
                    final['Acct_MName']=df.Data[9]
                elif t10=='Acct_LName':
                    final['Acct_LName']=df.Data[9]
                elif t10=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[9]
                elif t10=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[9]
                elif t10=='Acct_City':
                    final['Acct_City']=df.Data[9]
                elif t10=='Acct_State':
                    final['Acct_State']=df.Data[9]
                elif t10=='Acct_phone':
                    final['Acct_phone']=df.Data[9]
                elif t10=='Acct_email':
                    final['Acct_email']=df.Data[9]
                elif t10=='Acct_DOB':
                    final['Acct_DOB']=df.Data[9]
                elif t10=='Acct_Gender':
                    final['Acct_Gender']=df.Data[9]

                
                if t11=='Acct_id':
                    final['Acct_id']=df.Data[10]
                elif t11=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[10]
                elif t11=='Acct_FName':
                    final['Acct_FName']=df.Data[10]
                elif t11=='Acct_MName':
                    final['Acct_MName']=df.Data[10]
                elif t11=='Acct_LName':
                    final['Acct_LName']=df.Data[10]
                elif t11=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[10]
                elif t11=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[10]
                elif t11=='Acct_City':
                    final['Acct_City']=df.Data[10]
                elif t11=='Acct_State':
                    final['Acct_State']=df.Data[10]
                elif t11=='Acct_phone':
                    final['Acct_phone']=df.Data[10]
                elif t11=='Acct_email':
                    final['Acct_email']=df.Data[10]
                elif t11=='Acct_DOB':
                    final['Acct_DOB']=df.Data[10]
                elif t11=='Acct_Gender':
                    final['Acct_Gender']=df.Data[10]

                #st.dataframe(final)

                
            if len(ym1)==12:
                    
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5] 
                
                if t7=='Acct_id':
                    final['Acct_id']=df.Data[6]
                elif t7=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[6]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[6]
                elif t7=='Acct_MName':
                    final['Acct_MName']=df.Data[6]
                elif t7=='Acct_LName':
                    final['Acct_LName']=df.Data[6]
                elif t7=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[6]
                elif t7=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[6]
                elif t7=='Acct_City':
                    final['Acct_City']=df.Data[6]
                elif t7=='Acct_State':
                    final['Acct_State']=df.Data[6]
                elif t7=='Acct_phone':
                    final['Acct_phone']=df.Data[6]
                elif t7=='Acct_email':
                    final['Acct_email']=df.Data[6]
                elif t7=='Acct_DOB':
                    final['Acct_DOB']=df.Data[6]
                elif t7=='Acct_Gender':
                    final['Acct_Gender']=df.Data[6]

                if t8=='Acct_id':
                    final['Acct_id']=df.Data[7]
                elif t8=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[7]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[7]
                elif t8=='Acct_MName':
                    final['Acct_MName']=df.Data[7]
                elif t8=='Acct_LName':
                    final['Acct_LName']=df.Data[7]
                elif t8=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[7]
                elif t8=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[7]
                elif t8=='Acct_City':
                    final['Acct_City']=df.Data[7]
                elif t8=='Acct_State':
                    final['Acct_State']=df.Data[7]
                elif t8=='Acct_phone':
                    final['Acct_phone']=df.Data[7]
                elif t8=='Acct_email':
                    final['Acct_email']=df.Data[7]
                elif t8=='Acct_DOB':
                    final['Acct_DOB']=df.Data[7]
                elif t8=='Acct_Gender':
                    final['Acct_Gender']=df.Data[7]
                
                if t9=='Acct_id':
                    final['Acct_id']=df.Data[8]
                elif t9=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[8]
                elif t9=='Acct_FName':
                    final['Acct_FName']=df.Data[8]
                elif t9=='Acct_MName':
                    final['Acct_MName']=df.Data[8]
                elif t9=='Acct_LName':
                    final['Acct_LName']=df.Data[8]
                elif t9=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[8]
                elif t9=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[8]
                elif t9=='Acct_City':
                    final['Acct_City']=df.Data[8]
                elif t9=='Acct_State':
                    final['Acct_State']=df.Data[8]
                elif t9=='Acct_phone':
                    final['Acct_phone']=df.Data[8]
                elif t9=='Acct_email':
                    final['Acct_email']=df.Data[8]
                elif t9=='Acct_DOB':
                    final['Acct_DOB']=df.Data[8]
                elif t9=='Acct_Gender':
                    final['Acct_Gender']=df.Data[8]
                
                
                if t10=='Acct_id':
                    final['Acct_id']=df.Data[9]
                elif t10=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[9]
                elif t10=='Acct_FName':
                    final['Acct_FName']=df.Data[9]
                elif t10=='Acct_MName':
                    final['Acct_MName']=df.Data[9]
                elif t10=='Acct_LName':
                    final['Acct_LName']=df.Data[9]
                elif t10=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[9]
                elif t10=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[9]
                elif t10=='Acct_City':
                    final['Acct_City']=df.Data[9]
                elif t10=='Acct_State':
                    final['Acct_State']=df.Data[9]
                elif t10=='Acct_phone':
                    final['Acct_phone']=df.Data[9]
                elif t10=='Acct_email':
                    final['Acct_email']=df.Data[9]
                elif t10=='Acct_DOB':
                    final['Acct_DOB']=df.Data[9]
                elif t10=='Acct_Gender':
                    final['Acct_Gender']=df.Data[9]

                
                if t11=='Acct_id':
                    final['Acct_id']=df.Data[10]
                elif t11=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[10]
                elif t11=='Acct_FName':
                    final['Acct_FName']=df.Data[10]
                elif t11=='Acct_MName':
                    final['Acct_MName']=df.Data[10]
                elif t11=='Acct_LName':
                    final['Acct_LName']=df.Data[10]
                elif t11=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[10]
                elif t11=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[10]
                elif t11=='Acct_City':
                    final['Acct_City']=df.Data[10]
                elif t11=='Acct_State':
                    final['Acct_State']=df.Data[10]
                elif t11=='Acct_phone':
                    final['Acct_phone']=df.Data[10]
                elif t11=='Acct_email':
                    final['Acct_email']=df.Data[10]
                elif t11=='Acct_DOB':
                    final['Acct_DOB']=df.Data[10]
                elif t11=='Acct_Gender':
                    final['Acct_Gender']=df.Data[10]
                
                if t12=='Acct_id':
                    final['Acct_id']=df.Data[11]
                elif t12=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[11]
                elif t12=='Acct_FName':
                    final['Acct_FName']=df.Data[11]
                elif t12=='Acct_MName':
                    final['Acct_MName']=df.Data[11]
                elif t12=='Acct_LName':
                    final['Acct_LName']=df.Data[11]
                elif t12=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[11]
                elif t12=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[11]
                elif t12=='Acct_City':
                    final['Acct_City']=df.Data[11]
                elif t12=='Acct_State':
                    final['Acct_State']=df.Data[11]
                elif t12=='Acct_phone':
                    final['Acct_phone']=df.Data[11]
                elif t12=='Acct_email':
                    final['Acct_email']=df.Data[11]
                elif t12=='Acct_DOB':
                    final['Acct_DOB']=df.Data[11]
                elif t12=='Acct_Gender':
                    final['Acct_Gender']=df.Data[11]

                #st.dataframe(final)

                
        if len(ym1)==13:
                    
                if t1=='Acct_id':
                    final['Acct_id']=df.Data[0]
                elif t1=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[0]
                elif t1=='Acct_FName':
                    final['Acct_FName']=df.Data[0]
                elif t1=='Acct_MName':
                    final['Acct_MName']=df.Data[0]
                elif t1=='Acct_LName':
                    final['Acct_LName']=df.Data[0]
                elif t1=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[0]
                elif t1=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[0]
                elif t1=='Acct_City':
                    final['Acct_City']=df.Data[0]
                elif t1=='Acct_State':
                    final['Acct_State']=df.Data[0]
                elif t1=='Acct_phone':
                    final['Acct_phone']=df.Data[0]
                elif t1=='Acct_email':
                    final['Acct_email']=df.Data[0]
                elif t1=='Acct_DOB':
                    final['Acct_DOB']=df.Data[0]
                elif t1=='Acct_Gender':
                    final['Acct_Gender']=df.Data[0]

                if t2=='Acct_id':
                    final['Acct_id']=df.Data[1]
                elif t2=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[1]
                elif t2=='Acct_FName':
                    final['Acct_FName']=df.Data[1]
                elif t2=='Acct_MName':
                    final['Acct_MName']=df.Data[1]
                elif t2=='Acct_LName':
                    final['Acct_LName']=df.Data[1]
                elif t2=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[1]
                elif t2=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[1]
                elif t2=='Acct_City':
                    final['Acct_City']=df.Data[1]
                elif t2=='Acct_State':
                    final['Acct_State']=df.Data[1]
                elif t2=='Acct_phone':
                    final['Acct_phone']=df.Data[1]
                elif t2=='Acct_email':
                    final['Acct_email']=df.Data[1]
                elif t2=='Acct_DOB':
                    final['Acct_DOB']=df.Data[1]
                elif t2=='Acct_Gender':
                    final['Acct_Gender']=df.Data[1]
                
                if t3=='Acct_id':
                    final['Acct_id']=df.Data[2]
                elif t3=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[2]
                elif t3=='Acct_FName':
                    final['Acct_FName']=df.Data[2]
                elif t3=='Acct_MName':
                    final['Acct_MName']=df.Data[2]
                elif t3=='Acct_LName':
                    final['Acct_LName']=df.Data[2]
                elif t3=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[2]
                elif t3=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[2]
                elif t3=='Acct_City':
                    final['Acct_City']=df.Data[2]
                elif t3=='Acct_State':
                    final['Acct_State']=df.Data[2]
                elif t3=='Acct_phone':
                    final['Acct_phone']=df.Data[2]
                elif t3=='Acct_email':
                    final['Acct_email']=df.Data[2]
                elif t3=='Acct_DOB':
                    final['Acct_DOB']=df.Data[2]
                elif t3=='Acct_Gender':
                    final['Acct_Gender']=df.Data[2]
                
                if t4=='Acct_id':
                    final['Acct_id']=df.Data[3]
                elif t4=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[3]
                elif t4=='Acct_FName':
                    final['Acct_FName']=df.Data[3]
                elif t4=='Acct_MName':
                    final['Acct_MName']=df.Data[3]
                elif t4=='Acct_LName':
                    final['Acct_LName']=df.Data[3]
                elif t4=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[3]
                elif t4=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[3]
                elif t4=='Acct_City':
                    final['Acct_City']=df.Data[3]
                elif t4=='Acct_State':
                    final['Acct_State']=df.Data[3]
                elif t4=='Acct_phone':
                    final['Acct_phone']=df.Data[3]
                elif t4=='Acct_email':
                    final['Acct_email']=df.Data[3]
                elif t4=='Acct_DOB':
                    final['Acct_DOB']=df.Data[3]
                elif t4=='Acct_Gender':
                    final['Acct_Gender']=df.Data[3]
                
                if t5=='Acct_id':
                    final['Acct_id']=df.Data[4]
                elif t5=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[4]
                elif t5=='Acct_FName':
                    final['Acct_FName']=df.Data[4]
                elif t5=='Acct_MName':
                    final['Acct_MName']=df.Data[4]
                elif t5=='Acct_LName':
                    final['Acct_LName']=df.Data[4]
                elif t5=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[4]
                elif t5=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[4]
                elif t5=='Acct_City':
                    final['Acct_City']=df.Data[4]
                elif t5=='Acct_State':
                    final['Acct_State']=df.Data[4]
                elif t5=='Acct_phone':
                    final['Acct_phone']=df.Data[4]
                elif t5=='Acct_email':
                    final['Acct_email']=df.Data[4]
                elif t5=='Acct_DOB':
                    final['Acct_DOB']=df.Data[4]
                elif t5=='Acct_Gender':
                    final['Acct_Gender']=df.Data[4]
                
                if t6=='Acct_id':
                    final['Acct_id']=df.Data[5]
                elif t6=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[5]
                elif t6=='Acct_FName':
                    final['Acct_FName']=df.Data[5]
                elif t6=='Acct_MName':
                    final['Acct_MName']=df.Data[5]
                elif t6=='Acct_LName':
                    final['Acct_LName']=df.Data[5]
                elif t6=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[5]
                elif t6=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[5]
                elif t6=='Acct_City':
                    final['Acct_City']=df.Data[5]
                elif t6=='Acct_State':
                    final['Acct_State']=df.Data[5]
                elif t6=='Acct_phone':
                    final['Acct_phone']=df.Data[5]
                elif t6=='Acct_email':
                    final['Acct_email']=df.Data[5]
                elif t6=='Acct_DOB':
                    final['Acct_DOB']=df.Data[5]
                elif t6=='Acct_Gender':
                    final['Acct_Gender']=df.Data[5] 
                
                if t7=='Acct_id':
                    final['Acct_id']=df.Data[6]
                elif t7=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[6]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[6]
                elif t7=='Acct_MName':
                    final['Acct_MName']=df.Data[6]
                elif t7=='Acct_LName':
                    final['Acct_LName']=df.Data[6]
                elif t7=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[6]
                elif t7=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[6]
                elif t7=='Acct_City':
                    final['Acct_City']=df.Data[6]
                elif t7=='Acct_State':
                    final['Acct_State']=df.Data[6]
                elif t7=='Acct_phone':
                    final['Acct_phone']=df.Data[6]
                elif t7=='Acct_email':
                    final['Acct_email']=df.Data[6]
                elif t7=='Acct_DOB':
                    final['Acct_DOB']=df.Data[6]
                elif t7=='Acct_Gender':
                    final['Acct_Gender']=df.Data[6]

                if t8=='Acct_id':
                    final['Acct_id']=df.Data[7]
                elif t8=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[7]
                elif t7=='Acct_FName':
                    final['Acct_FName']=df.Data[7]
                elif t8=='Acct_MName':
                    final['Acct_MName']=df.Data[7]
                elif t8=='Acct_LName':
                    final['Acct_LName']=df.Data[7]
                elif t8=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[7]
                elif t8=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[7]
                elif t8=='Acct_City':
                    final['Acct_City']=df.Data[7]
                elif t8=='Acct_State':
                    final['Acct_State']=df.Data[7]
                elif t8=='Acct_phone':
                    final['Acct_phone']=df.Data[7]
                elif t8=='Acct_email':
                    final['Acct_email']=df.Data[7]
                elif t8=='Acct_DOB':
                    final['Acct_DOB']=df.Data[7]
                elif t8=='Acct_Gender':
                    final['Acct_Gender']=df.Data[7]
                
                if t9=='Acct_id':
                    final['Acct_id']=df.Data[8]
                elif t9=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[8]
                elif t9=='Acct_FName':
                    final['Acct_FName']=df.Data[8]
                elif t9=='Acct_MName':
                    final['Acct_MName']=df.Data[8]
                elif t9=='Acct_LName':
                    final['Acct_LName']=df.Data[8]
                elif t9=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[8]
                elif t9=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[8]
                elif t9=='Acct_City':
                    final['Acct_City']=df.Data[8]
                elif t9=='Acct_State':
                    final['Acct_State']=df.Data[8]
                elif t9=='Acct_phone':
                    final['Acct_phone']=df.Data[8]
                elif t9=='Acct_email':
                    final['Acct_email']=df.Data[8]
                elif t9=='Acct_DOB':
                    final['Acct_DOB']=df.Data[8]
                elif t9=='Acct_Gender':
                    final['Acct_Gender']=df.Data[8]
                
                
                if t10=='Acct_id':
                    final['Acct_id']=df.Data[9]
                elif t10=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[9]
                elif t10=='Acct_FName':
                    final['Acct_FName']=df.Data[9]
                elif t10=='Acct_MName':
                    final['Acct_MName']=df.Data[9]
                elif t10=='Acct_LName':
                    final['Acct_LName']=df.Data[9]
                elif t10=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[9]
                elif t10=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[9]
                elif t10=='Acct_City':
                    final['Acct_City']=df.Data[9]
                elif t10=='Acct_State':
                    final['Acct_State']=df.Data[9]
                elif t10=='Acct_phone':
                    final['Acct_phone']=df.Data[9]
                elif t10=='Acct_email':
                    final['Acct_email']=df.Data[9]
                elif t10=='Acct_DOB':
                    final['Acct_DOB']=df.Data[9]
                elif t10=='Acct_Gender':
                    final['Acct_Gender']=df.Data[9]

                
                if t11=='Acct_id':
                    final['Acct_id']=df.Data[10]
                elif t11=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[10]
                elif t11=='Acct_FName':
                    final['Acct_FName']=df.Data[10]
                elif t11=='Acct_MName':
                    final['Acct_MName']=df.Data[10]
                elif t11=='Acct_LName':
                    final['Acct_LName']=df.Data[10]
                elif t11=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[10]
                elif t11=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[10]
                elif t11=='Acct_City':
                    final['Acct_City']=df.Data[10]
                elif t11=='Acct_State':
                    final['Acct_State']=df.Data[10]
                elif t11=='Acct_phone':
                    final['Acct_phone']=df.Data[10]
                elif t11=='Acct_email':
                    final['Acct_email']=df.Data[10]
                elif t11=='Acct_DOB':
                    final['Acct_DOB']=df.Data[10]
                elif t11=='Acct_Gender':
                    final['Acct_Gender']=df.Data[10]
                
                if t12=='Acct_id':
                    final['Acct_id']=df.Data[11]
                elif t12=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[11]
                elif t12=='Acct_FName':
                    final['Acct_FName']=df.Data[11]
                elif t12=='Acct_MName':
                    final['Acct_MName']=df.Data[11]
                elif t12=='Acct_LName':
                    final['Acct_LName']=df.Data[11]
                elif t12=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[11]
                elif t12=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[11]
                elif t12=='Acct_City':
                    final['Acct_City']=df.Data[11]
                elif t12=='Acct_State':
                    final['Acct_State']=df.Data[11]
                elif t12=='Acct_phone':
                    final['Acct_phone']=df.Data[11]
                elif t12=='Acct_email':
                    final['Acct_email']=df.Data[11]
                elif t12=='Acct_DOB':
                    final['Acct_DOB']=df.Data[11]
                elif t12=='Acct_Gender':
                    final['Acct_Gender']=df.Data[11]
                
                
                if t13=='Acct_id':
                    final['Acct_id']=df.Data[12]
                elif t13=='Acct_UIDNo.':
                    final['Acct_UIDNo.']=df.Data[12]
                elif t13=='Acct_FName':
                    final['Acct_FName']=df.Data[12]
                elif t13=='Acct_MName':
                    final['Acct_MName']=df.Data[12]
                elif t13=='Acct_LName':
                    final['Acct_LName']=df.Data[12]
                elif t13=='Acct_Addr1':
                    final['Acct_Addr1']=df.Data[12]
                elif t13=='Acct_Addr2':
                    final['Acct_Addr2']=df.Data[12]
                elif t13=='Acct_City':
                    final['Acct_City']=df.Data[12]
                elif t13=='Acct_State':
                    final['Acct_State']=df.Data[12]
                elif t13=='Acct_phone':
                    final['Acct_phone']=df.Data[12]
                elif t13=='Acct_email':
                    final['Acct_email']=df.Data[12]
                elif t13=='Acct_DOB':
                    final['Acct_DOB']=df.Data[12]
                elif t13=='Acct_Gender':
                    final['Acct_Gender']=df.Data[12]
            
        st.dataframe(final)
            
         
def app():
  
    st.title("Model Selector")
    opt = st.radio("Models",('Select one','nli-roberta-base', 'stsb-roberta-base'))

    if opt == 'nli-roberta-base':
        
        st.write('You can use this framework to compute sentence / text embeddings for more than 100 languages.These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning.'
         'This can be useful for semantic textual similar, semantic search, or paraphrase mining.')

        bert(opt)
    elif opt=='stsb-roberta-base':
        st.write("This is other bert model with little different features")
        bert(opt)
    else:
        st.write("Select one of the models to test the dataset against the present dataset")
    
    
        