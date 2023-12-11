import os
import json
import requests
import random
from dotenv import load_dotenv
from PIL import Image
from random import sample
from typing import List, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()


def get_movie_details(movie_id: int) -> dict:
    """ use tmdbId for tmdb API """
    # get poster url
    headers = {'Accept': 'application/json'}
    payload = {'api_key': os.getenv('TMDB_API_KEY')}
    response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
    response = json.loads(response.text)
    base_url = response['images']['base_url'] + 'w185'

    try:
        # Query themoviedb.org API for movie poster url.
        movie_url = 'http://api.themoviedb.org/3/movie/{:}'.format(movie_id)
        response = requests.get(movie_url, params=payload, headers=headers)
        resp = json.loads(response.text)
        
        result = {
            'synopsis': resp.get('overview', 'Not Available'),
            'runtime': resp.get('runtime', 0),  # is in minutes
            'poster_url': base_url + resp.get('poster_path') if resp.get('poster_path') else base_url,
            'is_adult': resp.get('adult', False),
            'release_date': resp.get('release_date', '1900-01-01'),
            'movie_id': movie_id,
            'title': resp.get('title', ''),
            'status_ok': True
        }
        return result
    except Exception as e:
        print(e)
        return


def minutes2hours(x: int) -> str:
    if x < 60:
        return f"{x} mins"
    elif x == 60:
        return f"1 hour"
    else:
        hours = x // 60
        mins = x - (hours * 60)
        return f"{hours} hours {mins} mins"


def read_image(img_url: str) -> Image.Image:
    try:
        img = Image.open(requests.get(img_url, stream=True).raw)
        return img
    except:
        try:
            img = Image.open(img_url)
            return img   # np.asarray(img)
        except:
            return None


def make_recommendation(movie_id: int, 
                        df: pd.DataFrame, 
                        vectorizer: TfidfVectorizer, 
                        embeddings: Any,
                        topk: int=5,) -> List:
    """ write logic for recommendation 
    Given an input movie_id, this function should return a list of
    recommended tmdbIds.

    topk: how many recommendations (tmdbIds) to return
    """
    movie_idx = df[df['tmdbId']==movie_id].index[0]
    similarity = cosine_similarity(embeddings[movie_idx],embeddings)
    similarity_list = similarity.flatten().tolist()
    similarity_df = pd.DataFrame({'Id':df['tmdbId'], 'similarity':similarity_list})
    similarity_df = similarity_df.sort_values(by='similarity', ascending=False)
    similar_items = similarity_df['Id'][1:11].to_list()
    output = random.sample(similar_items, topk)
    
    return output


