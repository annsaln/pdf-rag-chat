import pickle, faiss
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout
import numpy as np
import os
import time
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()
api_key= os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def load_index_and_chunks():
    with open("vector_store/doc_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("vector_store/doc_index.faiss")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    return chunks, index, model

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    
    for attempt in range(3):
        try:
            chat_response = client.chat.complete(
                model=model,
                messages=messages
            )
            return (chat_response.choices[0].message.content)
        except HTTPError as e:
            if e.response.status_code == 429:
                print("⚠️ Rate limit hit. Retrying...")
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise e  # re-raise unexpected HTTP errors
        except (ConnectionError, Timeout) as e:
            print(f"🔌 Network error: {e}. Retrying...")
            time.sleep(2 ** attempt)     
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break

    return "Sorry, I couldn't get a response from Mistral. Please try again later."


def get_answer(query, index, chunks, model):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k=10)
    context = "\n".join([" ".join(chunks[i-5:i+5]) for i in I[0]])
    prompt = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query. Use the same language as in the query.
        Query: {query}
        Answer:
        """

    return run_mistral(prompt)
