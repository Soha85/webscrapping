import faiss
import re
import warnings
import pandas as pd
from docutils.nodes import document

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
    faiss_index = 0
    def __init__(self):
        self.articles['content'] = [row['title'] + " " + row['content'] for x, row in self.articles.iterrows()]
        self.articles['cleaned_text'] = [self.preprocess_text(x) for x in self.articles['content']]


    def prepare_data(self):
        for index, doc in self.articles.iterrows():
            # Split the document into chunks
            chunks = self.chunk_text(doc['cleaned_text'], chunk_size=50)
            self.corpus_chunks.extend(chunks)  # Add chunks to the corpus
            # Get embeddings for each chunk
            embeddings = [self.get_embeddings(chunk) for chunk in chunks]
            self.chunk_embeddings.extend(embeddings)

        # Convert chunk_embeddings to a NumPy array for efficient retrieval
        self.chunk_embeddings = np.vstack(self.chunk_embeddings)
        return "Chunking & Embedding Done"

    def save_embeddings_to_faiss(self):
        embedding_dim = model.config.hidden_size
        self.faiss_index = self.create_faiss_index(embedding_dim)
        # Process each question-context pair
        for question, context in zip(self.articles["title"],self.articles["content"]):
            # Combine question and context (as one block of text)
            combined_text = question + " " + context
            preprocessed_text = self.preprocess_text(combined_text)

            # Chunk the combined text (if necessary) and generate embeddings
            chunks = self.chunk_text(preprocessed_text)
            self.corpus_chunks.extend(chunks)

            for chunk in chunks:
                embedding = self.get_embeddings(chunk)
                self.chunk_embeddings.append(embedding)

        # Convert embeddings to NumPy array (FAISS requires float32 arrays)
        self.chunk_embeddings = np.vstack(self.chunk_embeddings).astype('float32')

        # Add embeddings to FAISS index
        self.faiss_index.add(self.chunk_embeddings)
        print(len(self.faiss_index))
        return "Chunking & Embedding Done"


    def preprocess_text(self,text):
        text = text.lower()  # Convert to lowercase
        text = text.replace('\\n', '')
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        # text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
        return text.strip()

    def create_faiss_index(self,embedding_dim):
        index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
        return index




    # Chunking: Split long documents into smaller chunks
    def chunk_text(self,text, chunk_size=50):
        words = text.split()
        # Create chunks of approximately chunk_size words
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]



    def get_embeddings(self,text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

    def retrieve_documents(self,query, top_k=1):
        query_embedding = self.get_embeddings(query)
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)
        # Get top_k similar chunks
        top_k_idx = np.argsort(similarities[0])[-top_k:][::-1]
        return [self.corpus_chunks[i] for i in top_k_idx], similarities[0][top_k_idx]

    def retrieve_documents_faiss(self,query, k=1):
        query_embedding = self.get_embeddings(query)
        distances, indices = self.faiss_index.search(query_embedding, k)
        results,scores = [],[]
        for i, idx in enumerate(indices[0]):
            results.append(self.corpus_chunks[idx])
            scores.append(distances[0][i])
        print(len(results))
        return results,scores



    def generate_text(self,query):
        TG = pipeline('text-generation', model='gpt2', batch_size=128)
        TG.model.config.pad_token_id = TG.model.config.eos_token_id
        return self.rag_generate_text(query,TG)

    def get_answer(self,query):
        QA = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
        QA.model.config.pad_token_id = QA.model.config.eos_token_id
        return self.rag_get_answer(query,QA)

    def rag_generate_text(self,query,llm):
        retrieved_docs,_ = self.retrieve_documents_faiss(query,1)
        if not retrieved_docs:
            return "No relevant documents found."
        context =  ' '.join(retrieved_docs)
        generated = llm(f"Query: {query}\nContext: {context}\nAnswer:", max_new_tokens=200, num_return_sequences=1)
        return str(generated[0]['generated_text'].split("Answer:")[1].strip())

    def rag_get_answer(self,query,llm):
        retrieved_docs,_ = self.retrieve_documents_faiss(query,1)
        print(len(self.faiss_index))
        print(len(retrieved_docs))
        if not retrieved_docs:
            return "No relevant documents found."
        context =  ' '.join(retrieved_docs)
        generated = llm(question=query,context=context)
        return generated

