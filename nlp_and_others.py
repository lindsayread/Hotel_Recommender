import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import re
import string
import nltk
import os
import pydeck as pdk
from sklearn.preprocessing import StandardScaler
from tb_merged import cos_nlp, cos_others

def run_nlp_and_others(txt, selected_features, n_results):
    '''
    Function that runs the text input from the user through the cos_nlp function to get a row of cosine similarities of the user's text to each hotel's reviews.
    Also takes the user's selected features and runs it through the cos_combined function to get a row of cosine similarities of the user's selections to each hotel's features.
    Places 3 times heavier weight on NLP (text similarity) portion than the selected features as to not have features outweigh the user's unique specifications.
    Takes the average of the selected features cosine similarity and 3*(text cosine similarity) to get an overall similarity matrix for the user. 
    Then selects the top n_results most similar hotels to the user and returns a map of the top n_results hotels, the name of the hotel(s), the phone number(s), website(s), rating, and other information based on the feature(s) selected (ie. number of restaurants nearby, top 3 restaurants nearby if "Restaurants Nearby" was selected)
    '''
    text_similarity = 3*(cos_nlp(txt))
    cos_combined = cos_others(selected_features)

    cos_combined['nlp_portion'] = text_similarity
    cos_combined['mean_similarity'] = (cos_combined[329] + cos_combined['nlp_portion'])/4

    top3hotels = cos_combined.nlargest(n_results, 'mean_similarity')
    idx_numbers = list(top3hotels.index.values)
    all_info = pd.read_csv('all_info_cleaned.csv')
    top3df = all_info.iloc[idx_numbers,:]

    tb_returned_cols = ['name_x','rating_x','website','international_phone_number']
    for i in selected_features:
        if i == 'restaurants':
            tb_returned_cols.append('top_3_restaurants')
            tb_returned_cols.append('num_restaurants')
        if i == 'tourist_attractions':
            tb_returned_cols.append('top_3_tourist_attract')
            tb_returned_cols.append('num_tourist_attract')
        if i == 'art_gallery':
            tb_returned_cols.append('top_3_art_galleries')
            tb_returned_cols.append('num_art_galleries')
        if i == 'gyms':
            tb_returned_cols.append('top_3_gyms')
            tb_returned_cols.append('num_gyms')
        if i == 'shopping_malls':
            tb_returned_cols.append('top_3_shopping')
            tb_returned_cols.append('num_shopping')
        if i == 'bars':
            tb_returned_cols.append('top_3_bars')
            tb_returned_cols.append('num_bars')
        if i == 'casinos':
            tb_returned_cols.append('top_casinos')
            tb_returned_cols.append('num_casinos')
        if i == 'supermarkets':
            tb_returned_cols.append('top_3_supermarkets')
            tb_returned_cols.append('num_supermarkets')
    tb_returned = top3df[tb_returned_cols]
    top3df['lon'] = top3df['lng']
    coords_df = top3df[['lat','lon','name_x']]
    reindexed = tb_returned.reset_index(drop=True)

    idx_numbers = list(reindexed.index.values)
    st.write(f'Your Top {n_results} Recommended Hotels Are:')

    covidLayer = pdk.Layer(
        "ScatterplotLayer",
        data=coords_df,
        pickable=False,
        opacity=0.3,
        stroked=True,
        filled=True,
        radius_scale=10,
        radius_min_pixels=5,
        radius_max_pixels=60,
        line_width_min_pixels=1,
        get_position=["lon", "lat"],
        get_fill_color=[252, 136, 3],
        get_line_color=[255,0,0],

    )
    text_layer = pdk.Layer("TextLayer", data=coords_df, get_position=['lon','lat'],
    get_text='name_x',get_color=[0,0,0,200],get_size=15,get_alignment_baseline='bottom',)
    r = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(latitude=20.6034, longitude=-105.2253, zoom=10),
    layers=[covidLayer,text_layer])
    st.pydeck_chart(r)

    for i in idx_numbers:
        st.write(f'**Hotel {i+1}:**',reindexed['name_x'].iloc[i])
        for j in tb_returned_cols:
            if j == 'rating_x':
                st.write('*Hotel Rating:*',reindexed['rating_x'].iloc[i])
            elif j == 'website':
                st.write('*Website:*',reindexed['website'].iloc[i])
            elif j == 'international_phone_number':
                st.write('*Phone Number:*',reindexed['international_phone_number'].iloc[i])
            elif j == 'top_3_restaurants':
                if reindexed['num_restaurants'].iloc[i]==60:
                    st.write('*Restaurants:* There are over',str(reindexed['num_restaurants'].iloc[i]),' restaurants within walking distance (800 m). The top rated restaurants nearby are: ',reindexed['top_3_restaurants'].iloc[i])
                elif reindexed['num_restaurants'].iloc[i]==1:
                    st.write('*Restaurants:* There is ', str(reindexed['num_restaurants'].iloc[i]), ' restaurant within walking distance (800 m): ',reindexed['top_3_restaurants'].iloc[i])
                else:
                    st.write('*Restaurants:* There are ',str(reindexed['num_restaurants'].iloc[i]),' restaurants within walking distance (800 m). The top rated restaurants nearby are: ',reindexed['top_3_restaurants'].iloc[i])
            elif j == 'top_3_tourist_attract':
                if reindexed['num_tourist_attract'].iloc[i]==0:
                    st.write('*Tourist Attractions:* There are no tourist attractions within walking distance (800 m).')
                elif reindexed['num_tourist_attract'].iloc[i]==1:
                    st.write('*Tourist Attractions:* There is ', str(reindexed['num_tourist_attract'].iloc[i]), ' tourist attraction within walking distance (800 m): ',reindexed['top_3_tourist_attract'].iloc[i])
                else:
                    st.write('*Tourist Attractions:* There are ', str(reindexed['num_tourist_attract'].iloc[i]), ' tourist attractions within walking distance (800 m). The top rated tourist attractions nearby are: ',reindexed['top_3_tourist_attract'].iloc[i])
            elif j == 'top_3_art_galleries':
                if reindexed['num_art_galleries'].iloc[i]==0:
                    st.write('*Art Galleries:* There are no art galleries within walking distance (800 m).')
                elif reindexed['num_art_galleries'].iloc[i]==1:
                    st.write('*Art Galleries:* There is ', str(reindexed['num_art_galleries'].iloc[i]), ' art gallery within walking distance (800 m): ',reindexed['top_3_art_galleries'].iloc[i])
                else:
                    st.write('*Art Galleries:* There are ', str(reindexed['num_art_galleries'].iloc[i]), ' art galleries within walking distance (800 m). The top rated art galleries nearby are: ',reindexed['top_3_art_galleries'].iloc[i])
            elif j == 'top_3_gyms':
                if reindexed['num_gyms'].iloc[i]==0:
                    st.write('*Gyms:* There are no gyms within walking distance (800 m).')
                elif reindexed['num_gyms'].iloc[i]==1:
                    st.write('*Gyms:* There is ', str(reindexed['num_gyms'].iloc[i]), ' gym within walking distance (800 m): ',reindexed['top_3_gyms'].iloc[i])
                else:
                    st.write('*Gyms:* There are ', str(reindexed['num_gyms'].iloc[i]), ' gyms within walking distance (800 m). The top rated gyms nearby are:',reindexed['top_3_gyms'].iloc[i])
            elif j == 'top_3_shopping':
                if reindexed['num_shopping'].iloc[i]==0:
                    st.write('*Shopping Malls:* There are no shopping malls within walking distance (800 m).')
                elif reindexed['num_shopping'].iloc[i]==1:
                    st.write('*Shopping Malls:* There is ', str(reindexed['num_shopping'].iloc[i]), ' shopping mall within walking distance (800 m): ',reindexed['top_3_shopping'].iloc[i])
                else:
                    st.write('*Shopping Malls:* There are ', str(reindexed['num_shopping'].iloc[i]), ' shopping malls within walking distance (800 m). The top rated shopping malls nearby are:',reindexed['top_3_shopping'].iloc[i])
            elif j == 'top_3_bars':
                if reindexed['num_bars'].iloc[i]==0:
                    st.write('*Bars:* There are no bars within walking distance (800 m).')
                elif reindexed['num_bars'].iloc[i]==1:
                    st.write('*Bars:* There is ', str(reindexed['num_bars'].iloc[i]), ' bar within walking distance (800 m): ',reindexed['top_3_bars'].iloc[i])
                else:
                    st.write('*Bars:* There are ', str(reindexed['num_bars'].iloc[i]), ' bars within walking distance (800 m). The top rated bars nearby are: ',reindexed['top_3_bars'].iloc[i])
            elif j == 'top_casinos':
                if reindexed['num_casinos'].iloc[i]==0:
                    st.write('*Casinos:* There are no casinos within walking distance (800 m).')
                elif reindexed['num_casinos'].iloc[i]==1:
                    st.write('*Casinos:* There is ', str(reindexed['num_casinos'].iloc[i]), ' casino within walking distance (800 m): ',reindexed['top_casinos'].iloc[i])
                else:
                    st.write('*Casinos:* There are ', str(reindexed['num_casinos'].iloc[i]), ' casinos within walking distance (800 m). Some casinos nearby are:',reindexed['top_casinos'].iloc[i])
            elif j == 'top_3_supermarkets':
                if reindexed['num_supermarkets'].iloc[i]==0:
                    st.write('*Supermarkets:* There are no supermarkets within walking distance.')
                elif reindexed['num_supermarkets'].iloc[i]==1:
                    st.write('*Supermarkets:* There is ', str(reindexed['num_supermarkets'].iloc[i]), ' supermarket within walking distance (800 m): ',reindexed['top_3_supermarkets'].iloc[i])
                else:
                    st.write('*Supermarkets:* There are ', str(reindexed['num_supermarkets'].iloc[i]), ' supermarkets within walking distance (800 m). Some supermarkets nearby are:',reindexed['top_3_supermarkets'].iloc[i])
    st.write('      ')

    return st.write('**Enjoy your customized stay in Puerto Vallarta!**')
