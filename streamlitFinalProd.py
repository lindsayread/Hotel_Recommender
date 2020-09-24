import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import re
import string
import nltk
from nlp_and_others import run_nlp_and_others
import time

st.title('Hotel/Lodging Recommender for Puerto Vallarta, Mexico')

st.subheader('Please describe your ideal hotel/lodging')

text = st.text_area('Example: I would like a luxurious hotel that is clean and quiet. It is important to me that the staff is friendly and attentive.')
n_results = st.slider('How many hotel/lodging results would you like returned to you?', 1,10,3)

st.sidebar.markdown('For the following places, please identify any that are important to you during your stay:')

restaurants = st.sidebar.checkbox('Restaurants Nearby')
tourist_attractions = st.sidebar.checkbox('Tourist Attractions Nearby')
art_gallery = st.sidebar.checkbox('Art Galleries Nearby')
gyms = st.sidebar.checkbox('Gyms Nearby')
shopping = st.sidebar.checkbox('Shopping Malls Nearby')
bars = st.sidebar.checkbox('Bars Nearby')
casinos = st.sidebar.checkbox('Casinos Nearby')
supermarkets = st.sidebar.checkbox('Supermarkets Nearby')

selected_features = []
if restaurants:
    selected_features.append('restaurants')
if tourist_attractions:
    selected_features.append('tourist_attractions')
if art_gallery:
    selected_features.append('art_gallery')
if gyms:
    selected_features.append('gyms')
if shopping:
    selected_features.append('shopping_malls')
if bars:
    selected_features.append('bars')
if casinos:
    selected_features.append('casinos')
if supermarkets:
    selected_features.append('supermarkets')

if st.button('Submit Request'):
    if len(text) <10:
        st.write('Please describe your ideal hotel/lodging in more detail')
    else:
        with st.spinner('Please be patient as we tailor your results (this could take up to 1 minute)'):
            time.sleep(5)
        run_nlp_and_others(text, selected_features, n_results)
else:
    st.write('Please submit text for request')
