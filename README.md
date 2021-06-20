# Masters-Dissertation
Analysing Retail Sector News using NLP during COVID-19. We live in an information era. We hear the news everywhere and see it. We open our social media account and our newsfeed contains news. We turn on your television and a news flash is aired. Newspapers here and there being sold out.  Let us talk about the news on Google. How often does one check Google news individually? Every nanosecond or less, it refreshes. So this implementation helps to find out the similarity of articles in less time.
# Getting Started
API’s are live now. They are updating everyday. I have extracted the data from one of news data collection website: https://webhose.io/. I have got 4 months of news for retail related keywords as seen below. Webhose uses specific querying technique to fetch JSON data.
# Prerequisites
Softwares that are required to execute 

Python - 3.6.4 Version 
IDE - PyCharm , Jupyter Notebook
# Installing

Follow the below steps one by one to run the application

Open the source code with the PyCharm IDE

Next Activate the Python virtual environment which you have configured

cd env/Scripts activate

Next install the required supporting packages using the Pip command

pip3 install -r requirements.txt 

Download the data which is generated from Webhose.

Next run the python flask application

python app.py 

Then open the URL in the brower which is running as a web instance.
# Data extraction
I have extracted the data from one of news data collection website: https://webhose.io/. I have got 4 months of news for retail related keywords as seen below. Webhose uses specific querying technique to fetch JSON data.
# Data Transformation
I got the data as 4 big json files. The Data transformation phase involves flattening the data from JSON files for each month to a single .CSV file.
# Data Cleaning
Cleaning of data is very important be'cause news articles contains an number of characters ranging from ASCII values to numerics, hexadecimals and special characters Etc.,
So here 1st I'll be doing the English Contractions like I will be replacing ain't too are not and appending into the text block 
•	Tokenisation: break the text into phrases and phrases into words; Lower the words and avoid punctuation. 
•	Delete words with less than 3 letters. 
•	They delete all the stop words. 
•	Words are lemmatized — third-person words are changed to first-person words, and verbs are converted in past and future tenses into present. 
•	Words are truncated — words are reduced to the root form.
# Exploratory Data Analysis
Exploratory data analysis (EDA) is a systematic way to explore the data using transformation and visualization. EDA is an iterative cycle and it’s not a process with any set of rules
# Train the model
python train.py 
# Test the model 
# Authors
*AMANI GOLI - 2944590 - Griffith College Dublin. - Masters in Big Data Management and Analytics.
# License
This project is open to use it for learning and make enhancements to the code without any restrictions.
# Acknowledgments
I owe my sincere gratitude and thanks to my professor and guide Dr Aqeel Kazmi for guiding me throughout the process of implementing and documenting
