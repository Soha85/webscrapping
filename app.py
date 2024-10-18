import logging
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from RAG import RAG
import numpy as np
import pandas as pd

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

articles =[ ]
# Function to scrape article URLs from a website
def scrape_articles(site_url):
    try:
        response = requests.get(site_url, timeout=10)  # Add a timeout for safety
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching the article: {e}"

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
    return title, full_text.strip()



# Route to render the main page with dropdown and article list
@app.route('/')
def index():
    return render_template('index.html', articles=[])

# Route to handle website selection and scrape article URLs
@app.route('/get_articles', methods=['POST'])
@app.route('/get_articles', methods=['POST'])
def get_articles():
    selected_website = request.form['website']
    article_links, titles, articles = [], [], []

    try:
        response = requests.get(selected_website, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/article/' in href:
                full_url = 'https://www.bbc.com' + href
                article_links.append(full_url)

        article_links = list(set(article_links))  # Remove duplicates

        for article in article_links:
            title, content = scrape_articles(article)
            if title and content:
                titles.append(title)
                articles.append(content)

        RAG.articles = pd.DataFrame({'title': titles, 'content': articles})

    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=f"Failed to fetch articles: {e}", articles=[])

    return render_template('index.html', articles=titles)


# Route to handle user question and send it to the Python script for processing

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form['question']
    response = "No Embedded Data"

    if not RAG.articles.empty:
        response = RAG().prepare_data(question)
    else:
        logger.error("No embedded data available for processing.")

    return render_template('index.html', answer=response)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

