from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from RAG import RAG
import numpy as np
import pandas as pd

app = Flask(__name__)
articles =[ ]
# Function to scrape article URLs from a website
def scrape_articles(site_url):
    response = requests.get(site_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text within the article (based on inspection)
    div_elements = soup.find_all('div', {'data-component': 'text-block'})
    all_paragraphs = []

    title = soup.title.get_text()

    for div in div_elements:
        paragraphs = div.find_all('p')
        for para in paragraphs:
            all_paragraphs.append(para.get_text())
    full_text = ' '.join(all_paragraphs)
    return title,full_text.strip()


# Route to render the main page with dropdown and article list
@app.route('/')
def index():
    return render_template('index.html', articles=[])

# Route to handle website selection and scrape article URLs
@app.route('/get_articles', methods=['POST'])
def get_articles():
    selected_website = request.form['website']

    response = requests.get(selected_website)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find article links (the exact class may change, inspect the page to adjust it)
    article_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/article/' in href:
            full_url = 'https://www.bbc.com' + href
            if full_url not in article_links:
                article_links.append(full_url)
    articles= []
    titles = []
    for article in article_links:
        title,content = scrape_articles(article)
        articles.append(content)
        titles.append(title)
    RAG.articles['content']=articles
    RAG.articles['title']=titles
    return render_template('index.html', articles=titles)

# Route to handle user question and send it to the Python script for processing
@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form['question']
    # Call a function to process the question and return the response

    response =""

    if(len(RAG.articles)>0):
        response = RAG().prepare_data(question)
    else:
        response = "No Embedded Data"
    return render_template('index.html', answer=response)



if __name__ == '__main__':
    app.run(debug=True)

