import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import re
import string
import nltk
import os
from sklearn.preprocessing import StandardScaler

def cos_nlp(text):
    '''
    Function that takes the user's text as input, cleans the text, tokenizes the text into separate words, and runs each word through the same Google News Word2Vec model as was used on the Hotel Reviews.
    The words become 300 Dimensional Vectors and then are averaged into one 300 Dimensional Vector. 
    This vector is then compared to all Hotel Reviews vectors to find the cosine similarities of the user's text to each hotel. 
    One row, representing the cosine similarity of the user's text to each hotel's reviews, is returned.
    '''
    vectors = pd.read_csv('nlp_vectors.csv')

    input_text = text

    remove_n = lambda x: re.sub('\\n',' ',x)
    remove_singlen = lambda x: re.sub('\n',' ',x)
    alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    input_text1 = remove_n(input_text)
    input_text1 = remove_singlen(input_text1)
    input_text1 = alphanumeric(input_text1)
    input_text1 = punc_lower(input_text1)
    remove_emoji = lambda x: emoji_pattern.sub(r'', x)
    input_text1 = remove_emoji(input_text1)

    stoplist = set('for a an it its very of the and to in'.split())
    tokenized = [word for word in input_text1.lower().split() if word not in stoplist]

    google_vec_file = '/Users/lindsayread/Downloads/GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)

    viable_words = []
    for word in tokenized:
        try:
            model[word]
            viable_words.append(word)
        except:
            pass
    viables = viable_words

    df_first = pd.DataFrame([model[viables[0]]])
    for word in viables[1:]:
        df_first = pd.concat([df_first, pd.DataFrame([model[f'{word}']])], ignore_index=True)
    df_mean = pd.DataFrame(df_first.mean())
    text_vector = df_mean.T

    text_vector.columns = vectors.columns
    vectors_all = vectors.append(text_vector, ignore_index=True)

    similarity_matrix = pd.DataFrame(cosine_similarity(vectors_all))

    text_similarity = similarity_matrix[329][:-1]

    return text_similarity

def cos_others(selected_features):
    '''
    Function that takes the user's selected feature(s) as input. 
    For each selected feature, it will add the mean rating and the number of that feature as distinguishable features to aggregate on. It will set the user's "number of features nearby" as the average for that column. It will set the user's "mean rating" of each feature as 5 (the highest possible value), assuming that they care about the feature(s) nearby being high quality.
    It then scales all values in the dataframe and creates a cosine similarity matrix.
    One row, representing the cosine similarity of the user's selected features to each hotel's selected features, is returned.
    '''
    numerical_cols = ['rating_x','user_ratings_total_x']
    tb_returned_cols = ['name_x','rating_x','website','international_phone_number']
    for i in selected_features:
        if i == 'restaurants':
            numerical_cols.append('rest_mean_rating')
            numerical_cols.append('num_restaurants')
            tb_returned_cols.append('top_3_restaurants')
        if i == 'tourist_attractions':
            numerical_cols.append('tourist_mean_rating')
            numerical_cols.append('num_tourist_attract')
            tb_returned_cols.append('top_3_tourist_attract')
        if i == 'art_gallery':
            numerical_cols.append('art_gallery_mean_rating')
            numerical_cols.append('num_art_galleries')
            tb_returned_cols.append('top_3_art_galleries')
        if i == 'gyms':
            numerical_cols.append('gyms_mean_rating')
            numerical_cols.append('num_gyms')
            tb_returned_cols.append('top_3_gyms')
        if i == 'shopping_malls':
            numerical_cols.append('shopping_mean_rating')
            numerical_cols.append('num_shopping')
            tb_returned_cols.append('top_3_shopping')
        if i == 'bars':
            numerical_cols.append('bars_mean_rating')
            numerical_cols.append('num_bars')
            tb_returned_cols.append('top_3_bars')
        if i == 'casinos':
            numerical_cols.append('num_casinos')
            tb_returned_cols.append('top_casinos')
        if i == 'supermarkets':
            numerical_cols.append('supermarkets_mean_rating')
            numerical_cols.append('num_supermarkets')
            tb_returned_cols.append('top_3_supermarkets')

    no_nulls = pd.read_csv('numerical_cols.csv')
    qual_only = no_nulls[numerical_cols]

    input_fields = {}
    for i in numerical_cols:
        if i == 'user_ratings_total_x':
            input_fields[i] = 432
        elif i =='rest_mean_rating':
            input_fields[i] = 4.5
        elif i == 'num_restaurants':
            input_fields[i] = 48
        elif i == 'num_tourist_attract':
            input_fields[i] = 9
        elif i == 'num_art_galleries':
            input_fields[i] = 16
        elif i == 'num_gyms':
            input_fields[i] = 9
        elif i == 'num_shopping':
            input_fields[i] = 7
        elif i == 'num_bars':
            input_fields[i] = 30
        else:
            input_fields[i] = 5.0

    with_input = qual_only.append(input_fields, ignore_index=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(with_input)
    scaled_df = pd.DataFrame(scaled)

    similarity_matrix_others = pd.DataFrame(cosine_similarity(scaled_df))

    input_row = similarity_matrix_others[329][:-1]
    input_row_df = pd.DataFrame(input_row)

    return input_row_df
