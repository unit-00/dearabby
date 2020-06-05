import requests 
from bs4 import BeautifulSoup
from pymongo import MongoClient
import pandas as pd

from Scraper import Scraper


mongo_client = MongoClient()
db = mongo_client.dearabby

archives_coll = db.archives
articles_coll = db.articles
cleaned_articles_coll = db.cleaned_articles


# Will return inserted data into mongo collection
scraper = Scraper(archives_coll, articles_coll, year_start=1991, year_end=2021)

# Clean and extract questions and answers
scraper.clean_articles(cleaned_articles_coll)

results = []
for articles in cleaned_articles_coll.find({}, {'_id': 0, 'extracted_qa': 1}):
    extracted_qa = articles.keys()
    
    for advice in articles[extracted_qa]:
        categories = advice['categories']
        question = advice['question']
        answer = advice['answer']
        
        results.append({'categories': categories,
                        'question': question,
                        'answer': answer})
        
df = pd.DataFrame(results, columns=['categories', 'question', 'answer'])

# Clean out advice without any questions
df = df[df['question'].str.len() > 0]

# Combine questions and answer as one string
df['question'] = df['question'].str.join(' ').str.replace('DEAR .*:', '')
df['answer'] = df['answer'].str.join(' ').str.replace('DEAR .*:', '')

df['qa'] = df['question'] + ' ' + df['answer']

# Export to csv file
df[['categories', 'qa']].to_csv('dearabby_qa.csv')
