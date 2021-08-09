import streamlit as st
import numpy as np
import pandas as pd
from pages import utils

def app():
    
    st.title("Upload Data File")
    fileuploade=st.file_uploader("Upload CSV",type=["csv","xlsx"])
    global Data
    
    
    if fileuploade is not None:
        
        file_details={"File Name":fileuploade.name,"File Type":fileuploade.type,"File Size":fileuploade.size}
            
        Data=pd.read_csv(fileuploade)
        #bert(df)
            
        source,Target=st.beta_columns(2)
        source.header("Source")
        Target.header("Target")
        col1,col2=st.beta_columns(2)
        Source=Data.columns.values.tolist()
        
            
        oSELECTED = col1.multiselect('Select',Source)
            
            

        n=pd.read_csv('TargetDataBasecsv.csv')
        target=n.columns.values.tolist()
            
        for p in target:
            col2.write(p)
        st.title("Select the column which has dates")
        col4,col5=st.beta_columns(2)
        
        datt=col4.radio("Select",Source)
            
            
    if st.button("Load Data"):
        ar=Data[Source[Source.index(datt)]]
        
        from datetime import datetime
        for dt in ar:
            datee=dt
            form=['%d-%m-%Y','%m-%d-%Y','%B %d,%Y','%m/%d/%Y','%d/%m/%Y','%d-%m-%y','%d.%m.%Y','%m.%d.%Y','%d.%m.%y','%m.%d.%y','%m-%d-%y','%d/%m/%y','%m/%d/%y','%Y/%m/%d']
            i=0
            while int(i)<=int(len(form)): 
                try:
                    date_object = datetime.strptime(datee,form[i])
                    g = pd.to_datetime(date_object, format='%d%m%y')
                    
                    Data[Source[Source.index(datt)]].replace(datee,g.date())
                    break
                except:
        
                    i=i+1
        # Raw data 
        with open("out.txt", "w") as f1:
            opte = repr(oSELECTED)
            f1.write(opte)
        #Data.to_csv('pages/data.csv', index=False)
            
    