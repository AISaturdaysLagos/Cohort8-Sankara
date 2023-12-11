import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import read_image, minutes2hours, make_recommendation, get_movie_details
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
                    min_width=width,
                    interactive=False,
                    show_download_button=False,
                    show_label=False,
                    container=False)
    mdown = gr.Markdown(f"""### {title} <br> Rating: {rating:.0f}% <br> Release year: {release_year:.0f} <br> Runtime: {runtime} <br> {genres} """)
    text = gr.TextArea(synopsis)
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


# read data
df = pd.read_csv('./data/movies_cleaned.csv')

# create embeddings
df['embed'] = (
    df['title'] 
    + ' ' + df['genres']
    + ' ' + df['tag']
)

# vectorizer
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(df['embed'])
TOPK = 4


def drop_down_click(dropdown_btn: Any):
    global df, vectorizer, embeddings

    mask = df['title'] == dropdown_btn
    movie_id = df.loc[mask, 'tmdbId'].values[0]
    image, mdown, text = show_one_movie(movie_id=movie_id, df=df)
    recommended_ids = make_recommendation(movie_id=movie_id,
                            df=df,
                            vectorizer=vectorizer,
                            embeddings=embeddings,
                            topk=TOPK)
    row = show_recommendations(movie_ids=recommended_ids,
                               df=df,
                               height=300,
                               width=300)
    return image, mdown, text, *row


demo = gr.Blocks(title='MovieSense')
with demo:
    gr.Markdown("# MovieSense")
    dropdown_btn = gr.Dropdown(choices=df['title'].values.tolist(),
                               interactive=True,
                                value="Search")

    with gr.Group():
        with gr.Row():
            img = gr.Image()
            with gr.Column():
                mdown = gr.Markdown()
                text = gr.TextArea(label='')

    with gr.Group():
        with gr.Row():
            outputs = [img, mdown, text]
            for i in range(TOPK):
                with gr.Column(min_width=250):
                    with gr.Group():
                        outputs.append(gr.Image())
                        outputs.append(gr.Markdown())
                        outputs.append(gr.TextArea(label=''))

    dropdown_btn.input(drop_down_click, inputs=dropdown_btn,
                     outputs=outputs)
    

if __name__ == "__main__":
    demo.launch(share=True)