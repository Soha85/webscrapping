import streamlit as st
import logging
import requests
from bs4 import BeautifulSoup
from RAG import RAG
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.session_state.articles_df = RAG.articles

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

# Session state for storing scraped data
if "articles_df" not in st.session_state:
    st.session_state.articles_df = pd.DataFrame(columns=["title", "content"])

# Dropdown to select website
selected_website = st.selectbox("Select a website to scrape", ['https://www.bbc.com/travel', 'https://www.bbc.com/culture'])
# **Initialize previous_website in session state**
if "previous_website" not in st.session_state:
    st.session_state.previous_website = None
# Session state for storing scraped data
if "articles_df" not in st.session_state:
    st.session_state.articles_df = pd.DataFrame(columns=["title", "content"])

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
            #st.write(RAG.articles)
        else:
            st.warning('No articles found.')

        st.session_state.articles_df = RAG.articles
        st.success("Articles successfully scraped!")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch articles: {e}")

# Display articles in a table (if any)
# Session state for storing scraped data
if not st.session_state.articles_df.empty and "articles_df" in st.session_state:
    st.write(st.session_state.articles_df)
else:
    st.info("No articles scraped yet.")

# Input for user question
question = st.text_input("Ask a question:")

# Placeholder for answer while processing
answer_placeholder = st.empty()

# Button to send the question for processing
if st.button('Ask Question'):
    if not st.session_state.articles_df.empty:
        # Show "Processing..." message
        answer_placeholder.write("**Processing your question...**")

        # Simulate processing time (replace with your actual RAG.prepare_data call)
        import time

        time.sleep(2)  # Simulate processing time

        # Replace with your actual RAG processing logic
        answer_placeholder.write(RAG().prepare_data())
        response = RAG().rag_generate(question)
        answer_placeholder.empty()  # Clear placeholder
        st.write(f"Answer: {response}")
    else:
        st.error("No articles available for processing.")


# Clear controls when selected website changes
if selected_website != st.session_state.previous_website:
    st.session_state.previous_website = selected_website
    question = ""
    answer_placeholder.empty()
    st.session_state.articles_df = RAG.articles.empty