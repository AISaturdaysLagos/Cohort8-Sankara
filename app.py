import random
import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import (read_image, 
                   minutes2hours, 
                   make_recommendation, 
                   get_movie_details, 
                   search_title, 
                   weighted_rating)

from typing import Any


def show_one_movie(movie_id: int,
                   df: pd.DataFrame,
                   height: int=400, 
                   width: int=400,):
    if isinstance(movie_id, str):
        movie_id = int(movie_id)

    movie_details = get_movie_details(movie_id)
    if movie_details is None:
        return None
    
    print(f'details: {movie_details}')

    try:
        img_path = movie_details.get('poster_url', '')
        img = read_image(img_path)
        synopsis = movie_details.get('synopsis', 'Not Available')
        runtime = movie_details.get('runtime', 0)
        runtime = minutes2hours(runtime)

        mask = df['tmdbId'] == movie_id
        release_year = df.loc[mask, 'year'].values[0]
        rating = df.loc[mask, 'percentage score'].values[0]
        genres = df.loc[mask, 'genres'].values[0]
        title = df.loc[mask, 'title'].values[0][:-6]
    except Exception as e:
        print(e)
        return None

    image = gr.Image(img, type='pil', 
                    height=height, 
                    width=width, 
                    #min_width=width,
                    interactive=False,
                    show_download_button=False,
                    show_label=False,
                    container=False
                    )
    mdown = gr.Markdown(f"""### {title} <br> Rating: {rating:.0f}% <br> Release year: {release_year:.0f} <br> Runtime: {runtime} <br> {genres} """,)
    text = gr.TextArea(synopsis, max_lines=7, label='')
    return image,  mdown, text


def show_recommendations(movie_ids: list,
                   df: pd.DataFrame,
                   height: int=400, 
                   width: int=400,):
    outputs = []
    for movie_id in movie_ids:
        img, mdown, txt = show_one_movie(movie_id=movie_id,
                    df=df,
                    height=height,
                    width=width,)
        outputs.append(img)
        outputs.append(mdown)
        outputs.append(txt)
    return outputs


def trending_movies(df: pd.DataFrame, length: int=25, min_score: float=60):
    trends = df[
    (df['year'] == 2023 ) &  
    (df['percentage score'] >= min_score) 
    ]
    trends=trends.sort_values(by=["num_of_reviews_per_movie", "rating"], ascending=False)
    trending = trends['tmdbId'][:length*2].to_list()
    trending = random.sample(trending, length)
    return trending


def top_rated_movies(df: pd.DataFrame, length: int=25, min_score: float=75):
    top= df[
    (df['percentage score'] >= min_score)
    ] 
    top= top.sort_values(by=[ 'rating','year'], ascending=False)
    top_rated =top['tmdbId'][:length].to_list()
    top_rated = random.sample(top_rated, length)
    return top_rated


# read data
df = pd.read_csv('./data/movies_cleaned.csv')

# adjust ratings
C = df['rating'].mean()
m = 50
df['percentage score'] = df.apply(lambda x: weighted_rating(x['rating'], x['num_of_reviews_per_movie'], C, m), axis=1)

# create embeddings
df['embed'] = (
    df['title'] 
    + ' ' + df['genres']
    + ' ' + df['tag']
)

# vectorizer
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(df['embed'])
TOPK = 10
LENGTH = 25

# title vectorizer to search for titles
title_vectorizer = TfidfVectorizer()
title_embeddings = title_vectorizer.fit_transform(df['title'][:-6])


def search_click(search_box: Any):
    global df, vectorizer, embeddings

    titles = search_title(query=search_box,
                        df=df,
                        vectorizer=title_vectorizer,
                        embeddings=title_embeddings)
    print(f'query: {search_box}, titles: {titles}')
    drop_btn = gr.Dropdown(choices=titles,
                            interactive=True,)
    return drop_btn, gr.Group(visible=True)


def drop_down_click(dropdown_btn: Any):
    global df, vectorizer, embeddings

    mask = df['title'] == dropdown_btn
    movie_id = df.loc[mask, 'tmdbId'].values[0]
    image, mdown, text = show_one_movie(movie_id=movie_id, df=df, 
                                        height=300, width=250)
    recommended_ids = make_recommendation(movie_id=movie_id,
                            df=df,
                            vectorizer=vectorizer,
                            embeddings=embeddings,
                            topk=TOPK)
    row = show_recommendations(movie_ids=recommended_ids,
                               df=df,
                               height=300,
                               width=250)
    return image, mdown, text, *row, gr.Group(visible=True), gr.Group(visible=True)


trending_mv = trending_movies(df=df, length=LENGTH)
top_rated = top_rated_movies(df=df, length=LENGTH)
demo = gr.Blocks(title='MovieSense')
with demo:
    gr.Markdown("# MovieSense")
    with gr.Tab("Trending"):
         with gr.Row():
            for i in range(LENGTH):
                with gr.Column(min_width=250):
                    with gr.Group():
                        with gr.Column():
                            show_one_movie(trending_mv[i],df=df)
    
    with gr.Tab("Top Rated"):
         with gr.Row():
            for i in range(LENGTH):
                with gr.Column(min_width=250):
                    with gr.Group():
                        with gr.Column():
                            show_one_movie(top_rated[i],df=df)
      
    with gr.Tab('Search'):
        with gr.Row():
            search_box = gr.Textbox(label='', scale=2)
            search_btn = gr.Button('Search', size='sm', scale=1)

        with gr.Group(visible=False) as group1:
            search_drop_btn = gr.Dropdown(choices=[''],
                                interactive=True)
            
        search_btn.click(search_click, inputs=search_box,
                    outputs=[search_drop_btn, group1])
        
        with gr.Group(visible=False) as gr1:
            with gr.Row():
                img = gr.Image()
                with gr.Column():
                    mdown = gr.Markdown()
                    text = gr.TextArea(label='')

        with gr.Group(visible=False) as gr2:
            with gr.Row():
                search_outputs = [img, mdown, text]
                for i in range(TOPK):
                    with gr.Column(min_width=250):
                        with gr.Group():
                            search_outputs.append(gr.Image())
                            search_outputs.append(gr.Markdown())
                            search_outputs.append(gr.TextArea(label=''))
    
    search_outputs.append(gr1)
    search_outputs.append(gr2)
    search_drop_btn.input(drop_down_click, inputs=search_drop_btn,
                    outputs=search_outputs)
        

if __name__ == "__main__":
    demo.launch()  # share=True)


