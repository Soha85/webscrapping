import re
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import BertTokenizer, BertModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')
llm = pipeline('text-generation', model='gpt2', batch_size=128)
llm.model.config.pad_token_id = llm.model.config.eos_token_id

class RAG:
    articles = pd.DataFrame([])
    def __init__(self):
        self.articles['content'] = [row['title'] + " " + row['content'] for x, row in self.articles.iterrows()]
        self.articles['cleaned_text'] = [self.preprocess_text(x) for x in self.articles['content']]


    def prepare_data(self,query):
        corpus_chunks = []
        chunk_embeddings = []

        for index, doc in self.articles.iterrows():
            # Split the document into chunks
            chunks = self.chunk_text(doc['cleaned_text'], chunk_size=50)
            corpus_chunks.extend(chunks)  # Add chunks to the corpus
            # Get embeddings for each chunk
            embeddings = [self.get_embedding(chunk) for chunk in chunks]
            chunk_embeddings.extend(embeddings)

        # Convert chunk_embeddings to a NumPy array for efficient retrieval
        chunk_embeddings = np.vstack(chunk_embeddings)

        return self.rag_generate_answer(query, corpus_chunks, chunk_embeddings)

    def preprocess_text(self,text):
        text = text.lower()  # Convert to lowercase
        text = text.replace('\\n', '')
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        # text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
        return text.strip()

    # Chunking: Split long documents into smaller chunks
    def chunk_text(self,text, chunk_size=50):
        words = text.split()
        # Create chunks of approximately chunk_size words
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Generate embeddings using BERT
    def get_embedding(self,text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        # Use the [CLS] token's embedding for the entire sentence
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    def retrieve_documents(self,query, corpus_chunks, chunk_embeddings, top_k=1):
        query_embedding = self.get_embedding(query)
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, chunk_embeddings)
        # Get top_k similar chunks
        top_k_idx = np.argsort(similarities[0])[-top_k:][::-1]
        return [corpus_chunks[i] for i in top_k_idx], similarities[0][top_k_idx]

    def rag_generate_answer(self,query, corpus_chunks, chunk_embeddings):
        # Retrieve relevant chunks
        retrieved_docs, scores = self.retrieve_documents(query, corpus_chunks, chunk_embeddings)
        # print(f"Retrieved Chunks: {retrieved_docs} with scores {scores}\n")

        # Combine reformulated query with retrieved chunks
        context = " ".join(retrieved_docs)

        # Generate a response using the LLM with the combined input
        # Set max_new_tokens to control the length of the generated part
        generated = llm(f"Query: {query}\nContext: {context}\nAnswer:", max_new_tokens=200, num_return_sequences=1)

        return str(generated[0]['generated_text'].split("Answer:")[1].strip())

