import faiss
import re
import warnings
import pandas as pd
from docutils.nodes import document
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import BertTokenizer, BertModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = SentenceTransformer('all-MiniLM-L6-v2')  # Example with SBERT
import rouge


class RAG:
    articles = pd.DataFrame([])
    corpus_chunks = []
    chunk_embeddings = []
    faiss_index = None
    def __init__(self):
        self.articles['all_content'] = [row['title'] + " " + row['content'] for x, row in self.articles.iterrows()]
        self.articles['cleaned_text'] = [self.preprocess_text(x) for x in self.articles['all_content']]


    def prepare_data(self,chunk_size,overlap):
        for context in self.articles["all_content"]:
            # Combine question and context (as one block of text)
            # Split the document into chunks
            chunks = self.chunk_text(context,chunk_size,overlap)
            self.corpus_chunks.extend(chunks)  # Add chunks to the corpus
            # Get embeddings for each chunk
            embeddings = [self.get_embeddings(chunk) for chunk in chunks]
            self.chunk_embeddings.extend(embeddings)

        # Convert chunk_embeddings to a NumPy array for efficient retrieval
        self.chunk_embeddings = np.vstack(self.chunk_embeddings)

        # Add embeddings to FAISS index
        # Ensure faiss_index is initialized before adding embeddings
        if self.faiss_index is None:
            self.faiss_index = self.create_faiss_index(self.chunk_embeddings.shape[1])
        self.faiss_index.add(self.chunk_embeddings)
        # Save the FAISS index to a file
        faiss.write_index(self.faiss_index, "faiss_index.bin")

        return "**Chunking & Embedding Done and Working on Retrieving Now........**"

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
    def chunk_text(self,text, chunk_size, overlap):
        words = text.split(' ')
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        return chunks

    def get_embeddings(self,text):
        return model.encode(text)


    def retrieve_documents_cosine(self,query, top_k):
        query_embedding = self.get_embeddings([query])
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)
        # Get top_k similar chunks
        top_k_idx = np.argsort(similarities[0])[-top_k:][::-1]
        return [self.corpus_chunks[i] for i in top_k_idx], similarities[0][top_k_idx]

    def retrieve_documents_faiss(self,query, k):
        self.faiss_index = faiss.read_index("faiss_index.bin")
        query_embedding = self.get_embeddings([query])
        distances, indices = self.faiss_index.search(query_embedding, k)
        results,scores = [],[]
        for i, idx in enumerate(indices[0]):
            results.append(self.corpus_chunks[idx])
            scores.append(distances[0][i])

        return results,scores

    def rag_generate(self,query,context,temperature):
        try:
            llm = pipeline('text-generation', model='gpt2', batch_size=128)
            #llm.model.config.pad_token_id = llm.model.config.eos_token_id
            generated = llm(f"Query: {query}\nContext: {context}\nAnswer:",max_new_tokens=200,temperature=temperature,num_return_sequences=1)
            # Create ROUGE evaluator
            evaluator = rouge.Rouge()

            # Evaluate summaries
            scores = evaluator.get_scores(context, generated[0]['generated_text'].split('Answer:')[1])
            return generated[0]['generated_text'].split('Answer:')[1],scores

        except Exception as e:
            print(f"Error generating text: {e}")
            return



