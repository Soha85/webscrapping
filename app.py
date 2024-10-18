import streamlit as st
import logging
import requests
from bs4 import BeautifulSoup
from RAG import RAG
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to scrape article URLs from a website
def scrape_articles(site_url):
    try:
        response = requests.get(site_url, timeout=10)  # Add a timeout for safety
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the article: {e}")
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

# Streamlit UI
st.title("Web Article Scraper")

# Dropdown to select website
selected_website = st.selectbox("Select a website to scrape", ['https://www.bbc.com/travel', 'https://www.bbc.com/culture'])

# Button to get articles
if st.button('Get Articles'):
    article_links = []
    titles = []
    articles = []

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

        if articles:
            # Display articles in a table
            RAG.articles = pd.DataFrame({'title': titles, 'content': articles})
            st.write(RAG.articles)
        else:
            st.warning('No articles found.')

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch articles: {e}")

# Input for user question
question = st.text_input("Ask a question:")

# Button to send the question for processing
if st.button('Ask Question'):
    if not RAG.articles.empty:
        response = RAG().prepare_data(question)
        st.write(f"Answer: {response}")
    else:
        st.error("No embedded data available for processing.")
