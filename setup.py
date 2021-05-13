import wget
import zipfile
import bz2
import os
import os.path
from os import path
import fasttext

def run_step_one():
    print("Downloading dataset...")
    if not path.isfile("archive.zip"):
        url = "https://storage.googleapis.com/kaggle-data-sets/1305/800230/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210513%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210513T110842Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=5abe3327f28adad398e58f70c72664d194f2bb7f72df332badd9a8a6981e5f6b1443dc70b901baf4956c46fdeddf70e9f2171869260dc983ed81294b55e49873e6fbef7bf9eccf58811f7509952728abda438e699041d27cfce1e86085fbf2f5d5470ab793dfa526e2351679337d404b7dcac283f2ad13e50c31d2515c1c9e6403feffdf63a132b1fb3f806fa5c763ed886542f0de6295b79d52402b5d485abe32f5e38bcd81be1ee68868fb2d1a6a231123511f7b51e84bc0b6e71afa5de5724cc4db061e3c9347c43645b676782c30600d9fdb3b88d3178b786e48ea1e9ae5b3c215c2e7b6437df79ce57f087dc9d158fcad88916f2e0379b4dca8bc79b1c4"
        z = wget.download(url)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

def run_step_two():
    print("Converting files to .txt files...")
    file_names = []

    if not path.exists("dataset"):
        with zipfile.ZipFile("archive.zip",'r') as f:
            f.extractall("./dataset")
            file_names = f.namelist()

    if not len(file_names) == 0:
        for file_name in file_names:
            with bz2.open("dataset/" + file_name, "r") as bz_f:
                name = ""
                if "test" in file_name:
                    name = "test.ft.txt"
                else:
                    name = "train.ft.txt"
                bytes_decoded = bz_f.read().decode("utf-8")
                text_file = open("dataset/" + name, "w")
                text_file.write(bytes_decoded)
                text_file.close
                os.remove("dataset/" + file_name)

    print("Conversion complete.")

def run_step_three():
    print("Training model...")
    model = fasttext.train_supervised("dataset/train.ft.txt")
    model.save_model("model/trained_review_model.bin")
    print("Model trained and saved.")