# python_exam
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/benskov95/python_exam/HEAD)

## Project name: Amazon Customer Review Analysis

## Description
This is our Python exam project, which uses a dataset consisting of Amazon customer reviews to train a model
to determine the rating of reviews based on their content, as well as other operations such as determining
the popularity of words in the various ratings, length of review relative to rating etc.

## List of used technologies
- nltk
- fasttext
- wordcloud
- csv
- pandas
- re
- bs4
- scipy

## Installation Guide
We have added a cell at the top of the main notebook file which installs all libraries that are needed to run the rest of the code.

## User Guide
Basically just run the cells from top to bottom. Specifically the 3rd cell has the option to download certain files we have uploaded if you don't want to train the model and generate the files yourself. This cell takes a long time to run if you do not download the files.

## Status

### What has been done
We have completed all tasks we created for ourselves, however, the task involving training a model to determine whether a review is positive or negative has been done slightly differently than planned. We have instead trained it to classify reviews based on the 1-5 star rating system, meaning it will attempt to determine which rating the review has based on its text content, and not whether it is just positive in tone or negative.

## List of challenges

1. Træne en model til at kunne gennemskue, om et review er positivt eller negativt baseret på tekstindholdet (classification & natural language processing)
2. Gruppere mest populære ord i positive og negative reviews, og visualisere det
3. Sammenligne den trænede model mod det rigtige datasæt, for at se hvor nøjagtig den er ift. Datasættet
4. Se om der er en sammenhæng mellem længde af review og rating
5. Fordi datasættets format er simpel, kan vi webscrape reviews fra andre sider, og bruge vores trænede model på dem for at se, om den kan genneskue Om reviewet er positivt eller negativt.
