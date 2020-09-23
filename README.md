# **Metis Project 5: Hotel Recommender for Puerto Vallarta, Mexico - Summer 2020**

## About:

For my Final Project at Metis, I combined NLP techniques with feature engineering and unsupervised learning to create a recommendation system for hotels in Puerto Vallarta, Mexico.

## Process:

I used the Google Places API to get information on over 600 hotels in and around Puerto Vallarta, Mexico, while simultaniously using the Google Translate API to translate all hotel reviews to English. 

Once I had the 5 most recent hotel reviews (in English) for each hotel, I used NLP preprocessing techniques to clean and tokenize the text. I then ran each word through the Google News Word2Vec Model to get a 300 dimensional vector based on its context and syntax (word embeddings). I then created a function to take the get the mean 300 dimensional vector for each hotel based on all of the vectored words in that document. 

Using the latitude and longitude of each hotel, I used the Google Places API, again, to extract even more information. I created a function that would return the number of [restaurants] within 800 m, the mean rating of [restaurants] within 800 m, and the top 3 rated [restaurants] within 800 m for several different types of places, including restaurants, tourist attractions, art galleries, gyms, and casinos.

### The App:



## Files:

In this project, you will find a PDF of my presentation, code files for the StreamLit App I created, and code files for my data acquisition, pre-processing, and modeling.

### Slideshow PDF:

[PV_Trip_Recommender.pdf](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/PV_Trip_Recommender.pdf)

### Jupyter Notebook Files:

In [HotelsDF.ipynb](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/HotelsDF.ipynb), I used the Google Places API to get information on hotels in Puerto Vallarta.

In [EDA.ipynb](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/EDA.ipynb), I did some basic EDA and converted the JSON formatted geometry column to separate latitude and longitude columns.

In [NLPHotelReviews.ipynb](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/NLPHotelReviews.ipynb), I tried out some preliminary NLP topic modeling methods like NMF and LSA.

In [NLPforAPP.ipynb](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/NLPforAPP.ipynb), I used the Google News Word2Vec model with the Gensim library to vectorize all words for each hotel's reviews. I then got the mean of all vectorized words for each hotel to produce one, 300 dimensional vector for each hotel.

In RadiusAroundHotel.ipynb](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/RadiusAroundHotel.ipynb), I used the latitude and longitude of each hotel to call upon the Google Places API, again, to get information on places around the hotel (ie. restaurants, tourist attractions, art galleries, gyms, casinos, etc.). This notebook contains the functions I used to get information like: number of restaurants nearby, mean rating of restaurants nearby, and top 3 restaurants nearby.

In [StreamLitPrep.ipynb](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/StreamLitPrep.ipynb), I got the dataframes ready for use in the StreamLit app.

### Streamlit App Files:

In [streamlitFinalProd.py](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/streamlitFinalProd.py), you will find the basic layout of the StreamLit app.

[tb_merged.py](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/tb_merged.py) contains a function that return the cosine similarity matrix for the NLP (text input) portion of the app and a function that returns the cosine similarity matrix for the features selected portion of the app.

[nlp_and_others.py](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/nlp_and_others.py) calls on the functions from tb_merged to create a mean cosine similarity matrix (based on both the NLP (text input) cosine similarity matrix and the other selected features similarity matrix). This file also streamlines the output for the app, including a map of the recommended hotels, and information about each recommended hotel.

### README.md:
- [README.md](https://github.com/thisismetis/chi20_ds15/tree/master/student_submissions/projects/project-05/read_lindsay/README.md)

## Data:

Data for this project was gathered using the [Google Places API](https://developers.google.com/places/web-service/overview).

## Skills & Analysis:

- API Utilization
- Natural Language Processing (NLP) techniques
- Feature Engineering
- Unsupervised Learning Techniques
- Tokenization
- Word Embeddings
- Cosine Similarity
- Interactive StreamLit App
- Dashboard creation
