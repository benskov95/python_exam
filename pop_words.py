import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def prepare_category_dfs(csv_path):
    main_df = pd.read_csv(csv_path)
    # Adding column names since data doesn't have any by default
    main_df.columns = ["Rating", "Title", "Review"]

    # Creating dataframe for each rating
    df_list = []
    for x in tqdm(range(1, 6)):
        rating_df = main_df[main_df["Rating"] == x]
        df_list.append(rating_df)

    return df_list[0], df_list[1], df_list[2], df_list[3], df_list[4]

def clean_reviews(df):
    # Get raw strings to process and convert to cleaned lists of words
    reviews = df["Review"].values
    r_stopwords = stopwords.words('english')
    cleaned_reviews = []

    # words we noticed many of in the reviews that don't say much about the tone of the review
    extras = ["even", "think", "seem", "though"]
    r_stopwords.extend(extras)

    for review in tqdm(reviews):
        pure_text = scrub_words(review)
        tokens = word_tokenize(pure_text)
        lcase_tokens = list(map(str.lower,tokens))
        review_words = [w for w in lcase_tokens if not w in r_stopwords]
        cleaned_reviews.append(review_words)

    return cleaned_reviews

def scrub_words(text):
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()
    return text