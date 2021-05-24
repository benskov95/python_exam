import csv
import urllib.request
import tarfile
from os import path
import shutil
import pandas as pd
from scipy import stats
import numpy as np

def test():
    print("There is whole through")
    print(path)
    if(path.isfile("amazon_review_full_csv.tar.gz")):
        print("sss")
    else:
        print("aaa")

def download_dataset():
    print("Downloading dataset...")
    if not path.isfile("amazon_review_full_csv.tar.gz"):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/selfiecircleweb.appspot.com/amazon_review_full_csv.tar.gz", "amazon_review_full_csv.tar.gz")

        print("Download complete.")
    else:
        print("Dataset already downloaded.")


def extract_dataset():
    print("Extracting dataset...")

    if not path.exists("dataset"):
        # Extract files
        file = tarfile.open('amazon_review_full_csv.tar.gz')
        file.extractall('./dataset')
        file.close()

        # move to dataset folder and delete original folder
        shutil.move(path.join("./dataset/amazon_review_full_csv", "test.csv"), "./dataset")
        shutil.move(path.join("./dataset/amazon_review_full_csv", "train.csv"), "./dataset")
        shutil.rmtree("./dataset/amazon_review_full_csv")

    print("Extracting complete.")

def cleanOutlier(rating, df):
        review = df.loc[df['rating'] == rating]

        my_series = review["review"].squeeze()

        df = pd.DataFrame({'data':my_series})

        df['z_score'] = stats.zscore(df['data'])

        newdf = df.loc[df['z_score'].abs()<=3]

        return(newdf["data"].mean()) 

def doWork():

    ratingList = []
    reviewList = []

    with open(f"dataset/train.csv", encoding="utf8") as f:
            reader = csv.reader(f,delimiter=',')
            linecount = 0
            for row in reader:
                
                ratingList.append(row[0])
                reviewList.append(len(row[2]))


    d = {'rating': ratingList, 'review': reviewList}
    df = pd.DataFrame(data=d)

    for x in range(1,6):
        print("Ratings: " + str(x))
        print(cleanOutlier(str(x), df))