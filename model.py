from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import shutil
import os
from os import path
import fasttext
import urllib.request
import tarfile
import csv
from termcolor import colored
import nltk
import re
import bs4
import requests
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_setence(setence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def scrub_words(text):
        # remove html markup
        text = re.sub("(<.*?>)", "", text)

        # remove non-ascii and digits
        text = re.sub("(\\W|\\d)", "", text)

        # remove whitespace
        text = text.strip()
        return text
        # function to split text into word

    tokens = word_tokenize(setence)

    # Lower case
    tokens = list(map(str.lower, tokens))

    # Remove stop words
    tokens = [w for w in tokens if not w in stop_words]

    # Clean word
    tokens = [scrub_words(w) for w in tokens]

    # lemmatize
    tokens = [lemmatizer.lemmatize(word=word, pos='v') for word in tokens]

    return " ".join(tokens)


def clean_dataset():
    print("Cleaning traning and test data and converting to fasttext compatible files...")
    files = ["test", "train"]

    newArray = []

    for file in files:
        with open(f"dataset/{file}.csv", encoding="utf8") as f:
            reader = csv.reader(f)
            for row in reader:
                row.pop(1)

                setence = clean_setence(row[1])

                newArray.append("__label__" + row[0] + " " + setence)

        with open(f"dataset/{file}.txt", "w", encoding="utf-8") as txt_file:
            for line in newArray:
                txt_file.write(line + "\n")

    print("Done creating cleaned fasttext files.")


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
        shutil.move(os.path.join(
            "./dataset/amazon_review_full_csv", "test.csv"), "./dataset")
        shutil.move(os.path.join(
            "./dataset/amazon_review_full_csv", "train.csv"), "./dataset")
        shutil.rmtree("./dataset/amazon_review_full_csv")

    print("Extracting complete.")


def train_autotune(seconds):
    print("Training model...")
    model = fasttext.train_supervised(
        "dataset/train.txt", autotuneValidationFile='dataset/test.txt', autotuneDuration=seconds)
    print("Model trained, now saving...")
    model.save_model("model/trained_review_model.bin")
    print("Model saved.")


def test_model(model):
    print("Testing model...")
    model = fasttext.load_model(f"model/{model}")
    test = model.test("dataset/test.txt")
    print(test)


def download_model_with_90_precision():
    if not path.isfile("model/trained_review_model_0.90.bin"):
        print("Downloading 0.90 precision model...")

        if not os.path.isdir("model"):
            os.makedirs("model")

        urllib.request.urlretrieve(
            "https://storage.googleapis.com/selfiecircleweb.appspot.com/trained_review_model_0.90.bin", "model/trained_review_model_0.90.bin")

        print("Download complete.")
    else:
        print("Model already downloaded.")


def predict_with_model(reviews, model, printLines):
    print(f"Predicting {len(reviews)} reviews...")
    model = fasttext.load_model(f"model/{model}")

    correctCount = 0
    for i, review in enumerate(reviews):
        setence = clean_setence(review[0])
        prediction = model.predict(setence, k=5)

        star = prediction[0][0][-1]
        precision = "{:.2%}".format(prediction[1][0])
        star2 = prediction[0][1][-1]
        precision2 = "{:.2%}".format(prediction[1][1])

        color = "red"
        if review[1] == int(star):
            color = "green"
            correctCount += 1
        if(printLines):
            print(
                colored(
                    f"{i}. Corrent rating: {review[1]} Guess: Rating: {star} Probability: {precision}", color=color))

            color = "green" if review[1] == int(star2) else "red"
            print(
                colored(
                    f"{i}. Corrent rating: {review[1]} Guess: Rating: {star2} Probability: {precision2}", color=color))

    print("Results: ")
    print(f"Correct: {correctCount} - {correctCount / (len(reviews)) * 100}%")

    def calculateWrongPercentage():
        try:
            return ((len(reviews) - correctCount) / len(reviews)) * 100
        except ZeroDivisionError:
            return 0

    wrongPercent = calculateWrongPercentage()
    print(
        f"Wrong: {len(reviews) - correctCount} - {wrongPercent}%")


def get_model_args(model):
    print("Getting model arguments...")
    model = fasttext.load_model(f"model/{model}")
    args_obj = model.f.getArgs()
    for hparam in dir(args_obj):
        if not hparam.startswith('__'):
            print(f"{hparam} -> {getattr(args_obj, hparam)}")
    print("Done printing arguments")
