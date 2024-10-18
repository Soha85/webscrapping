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


class RAG:
    articles = pd.DataFrame([])
    corpus_chunks = []
    chunk_embeddings = []
    def __init__(self):
        self.articles['content'] = [row['title'] + " " + row['content'] for x, row in self.articles.iterrows()]
        self.articles['cleaned_text'] = [self.preprocess_text(x) for x in self.articles['content']]


    def prepare_data(self):
        for index, doc in self.articles.iterrows():
            # Split the document into chunks
            chunks = self.chunk_text(doc['cleaned_text'], chunk_size=50)
            self.corpus_chunks.extend(chunks)  # Add chunks to the corpus
            # Get embeddings for each chunk
            embeddings = [self.get_embedding(chunk) for chunk in chunks]
            self.chunk_embeddings.extend(embeddings)

        # Convert chunk_embeddings to a NumPy array for efficient retrieval
        self.chunk_embeddings = np.vstack(self.chunk_embeddings)
        return "Chunking & Embedding Done"

    def generate_text(self,query):
        TG = pipeline('text-generation', model='gpt2', batch_size=128)
        TG.model.config.pad_token_id = TG.model.config.eos_token_id
        return self.rag_generate_text(query,TG)

    def get_answer(self,query):
        QA = pipeline('question-answering', model='gpt2')
        QA.model.config.pad_token_id = QA.model.config.eos_token_id
        return self.rag_get_answer(query,QA)

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

    def retrieve_documents(self,query, top_k=1):
        query_embedding = self.get_embedding(query)
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)
        # Get top_k similar chunks
        top_k_idx = np.argsort(similarities[0])[-top_k:][::-1]
        return [self.corpus_chunks[i] for i in top_k_idx], similarities[0][top_k_idx]

    def rag_generate_text(self,query,llm):
        # Retrieve relevant chunks
        retrieved_docs, scores = self.retrieve_documents(query)
        # print(f"Retrieved Chunks: {retrieved_docs} with scores {scores}\n")

        # Combine reformulated query with retrieved chunks
        context = " ".join(retrieved_docs)

        # Generate a response using the LLM with the combined input
        # Set max_new_tokens to control the length of the generated part
        generated = llm(f"Query: {query}\nContext: {context}\nAnswer:", max_new_tokens=200, num_return_sequences=1)

        return str(generated[0]['generated_text'].split("Answer:")[1].strip())

    def rag_get_answer(self,query,llm):
        # Retrieve relevant chunks
        retrieved_docs, scores = self.retrieve_documents(query)
        # print(f"Retrieved Chunks: {retrieved_docs} with scores {scores}\n")

        # Combine reformulated query with retrieved chunks
        context = " ".join(retrieved_docs)

        # Generate a response using the LLM with the combined input
        # Set max_new_tokens to control the length of the generated part
        generated = llm(question=query,context=context)

        return str(generated[0]['generated_text'].split("Answer:")[1].strip())

