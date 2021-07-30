import streamlit as st
import pandas as pd
Modules=["Home","Dataset Upload","Model selector","Results","Review & Download"]
selectbox = st.sidebar.selectbox("Modules",Modules)
if selectbox==Modules[0]:
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
elif selectbox==Modules[1]:
    st.header("Upload Data File")
    fileuploade=st.file_uploader("Upload CSV",type=["csv"])
    if fileuploade is not None:
        
        file_details={"File Name":fileuploade.name,
        "File Type":fileuploade.type,"File Size":fileuploade.size}
        st.write(file_details)
        df=pd.read_csv(fileuploade)
        print(fileuploade.name)
        st.dataframe(df)
elif selectbox==Modules[2]:
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
elif selectbox==Modules[3]:
    pass
elif selectbox==Modules[4]:
    pass