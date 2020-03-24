import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def get_title_from_index(index):
        ##????
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]



##Step 1: Read CSV File
df=pd.read_csv("movie_dataset.csv")
#print(df.head())
#print(df.columns)


##Step 2: Select Features
"""
(['index', 'budget', 'genres', 'homepage', 'id', 'keywords',
       'original_language', 'original_title', 'overview', 'popularity',
       'production_companies', 'production_countries', 'release_date',
       'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'title',
       'vote_average', 'vote_count', 'cast', 'crew', 'director'],
      dtype='object')
"""

features=["keywords","cast","genres","director"]



##Step 3: Create a column in DF which combines all selected features
for feature in features:
        #replaces all columns having NAN with empty sring
        df[feature]=df[feature].fillna("")



def combine_features(row):
        try:
                return row["keywords"]+" "+row["cast"]+" "+row["genres"]+" "+row["director"]
        #exception created due to "NaN"
        except:
                print ("Error:",row)
                
#to pass each row of the dataset individually to the function
df["combined_features"]=df.apply(combine_features,axis=1)
#print("Combined features:",df["combined_features"].head())



##Step 4: Create count matrix from this new combined column
cv=CountVectorizer()

count_matrix=cv.fit_transform(df["combined_features"])


##Step 5: Compute the Cosine Similarity based on the count_matrix

cosine_sim=cosine_similarity(count_matrix)

#matrix->  #movies * #movies

#print(cosine_sim)
movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title
movie_index=get_index_from_title(movie_user_likes)

#[1,0.8,0.2,0.5]-->[(0,1),(1,0.8),(2,0.2),(3,0.5)]
similar_movies=list(enumerate(cosine_sim[movie_index]))
#print(similar_movies)


## Step 7: Get a list of similar movies in descending order of similarity score

#sort accordinf to second element of tuple and in descending order
sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

## Step 8: Print titles of first 50 movies

for movie in sorted_similar_movies[:50]:
        print(get_title_from_index(movie[0]))
