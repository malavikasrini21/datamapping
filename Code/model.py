import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
def app():
    global opti
    opti=''
    st.title("Model Selector")
    opt = st.radio("Models",('Select one','nli-roberta-base', 'stsb-roberta-base'))

    if opt == 'nli-roberta-base':
        opti='nli-roberta-base'
        st.write('You can use this framework to compute sentence / text embeddings for more than 100 languages.These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning.'
         'This can be useful for semantic textual similar, semantic search, or paraphrase mining.')
    elif opt=='stsb-roberta-base':
        opti='stsb-roberta-base'
        st.write("This is other")
    else:
        st.write("Select one of the models to test the dataset against the present dataset")
    
    with open("out.txt", "w") as f1:
        opte = repr(opti)
        f1.write(opte)

        