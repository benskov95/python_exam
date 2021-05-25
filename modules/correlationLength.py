import csv
import pandas as pd
from scipy import stats

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
            for row in reader:
                
                ratingList.append(row[0])
                reviewList.append(len(row[2]))


    d = {'rating': ratingList, 'review': reviewList}
    df = pd.DataFrame(data=d)

    for x in range(1,6):
        print("Ratings: " + str(x))
        print(cleanOutlier(str(x), df))