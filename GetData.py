import pandas as pd

comments_df = pd.read_csv('commentsWithSenitiments.csv')


def get_random_comment():
    return comments_df[(comments_df['pol'] > 0.5) | (comments_df['pol'] < -0.5)].sample(1)['comment_text'].values[0]