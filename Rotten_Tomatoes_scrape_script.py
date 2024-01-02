# -*- coding: utf-8 -*-
"""
In this script we navigate to the Rotten Tomatoes page (https://www.rottentomatoes.com/browse/movies_at_home/affiliates:netflix~sort:critic_highest) where we scrape the data (movie name, 
streaming date, audience score and critics score) for the most popular movies for various platforms and store them into a Pandas data frame.
"""

# import requred libraries and modules
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Define a function that takes one string parameter called `platform` and allows you to modify the URL of the web page in order to return a dataframe containing the top movies from that 
# platform (make sure to add a `platform` column that specifies which platform the data came from).
def get_movies(platform):
    # get source page
    base_url = 'https://www.rottentomatoes.com/browse/movies_at_home/affiliates:platform,netflix~sort:critic_highest'
    url = base_url.replace('platform', platform)
    page = requests.get(url, headers={'Accept-Language': "lang=en-US"})
    soup = BeautifulSoup(page.content, "html.parser")
    movies = soup.find_all(class_='js-tile-link')

    # get movie information
    movie_names = []    
    movie_dates = []
    movie_scores_audience = []
    movie_scores_critics = []

    # access the appropriate information depending on the tag used
    for m in movies: 
        movie_names.append(m.find_all('span')[-2].text.strip())
        movie_dates.append(m.find_all('span')[-1].text.strip())
        movie_scores_audience.append(m.find('score-pairs-deprecated').attrs.get('audiencescore'))
        movie_scores_critics.append(m.find('score-pairs-deprecated').attrs.get('criticsscore'))

    # create data frame
    df = pd.DataFrame(
        {'platform': str(platform),
        'movie_name': movie_names,
        'streaming_date': movie_dates,
        'audience_score': movie_scores_audience, 
        'critics_score': movie_scores_critics
        })

    # clean data frame
    df['streaming_date'] = pd.to_datetime(df['streaming_date'].str.replace('Streaming ', ''))
    df.loc[df['audience_score']=='', 'audience_score'] = np.NaN
    df = df.astype({'audience_score':'float', 'critics_score':'float'})
    df.sort_values(['critics_score', 'audience_score'], ascending = False, inplace = True, ignore_index=True)

    return df

# Append the data frames generated by the `get_movies()` function for all your favorite platforms.

# Initiate an empty pandas data frame
    df_platform = pd.DataFrame(
        {'platform': [],
        'movie_name': [],
        'streaming_date': [],
        'audience_score': [], 
        'critics_score': []
        })
# Use a for loop to append the data frames generated by the `get_movies()` to the empty data frame
for p in (['netflix', 'amazon_prime', 'paramount_plus']):
    df_temp = get_movies(p)
    df_platform = pd.concat([df_platform, df_temp])
# re-index the data frame to avoid having duplicates in the index
df_platform.index = pd.RangeIndex(len(df_platform.index))
df_platform.sample(10)