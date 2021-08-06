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
        
            
        for p in Source:
            sav=col1.checkbox(p)
            
            

            n=pd.read_csv('TargetDataBasecsv.csv')
            target=n.columns.values.tolist()
            
        for p in target:
            col2.write(p)
        st.title("Select the column which has dates")
        col4,col5=st.beta_columns(2)
        
        for g in Source:
            col4.checkbox(g,key='dat')
            
            
    if st.button("Load Data"):
        
        # Raw data 
        with open("out.txt", "w") as f1:
            opte = repr(Source)
            f1.write(opte)
        #Data.to_csv('pages/data.csv', index=False)
            
    