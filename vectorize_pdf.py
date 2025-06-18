from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import sys

pdfpath = sys.argv[1]

def extract_pdf_elements(pdf_path):
    elements = partition_pdf(pdf_path)
    texts = [el.text.strip() for el in elements if el.text and len(el.text.strip()) > 0]
    return texts

# recursive chunking by characters
# TODO: test other methods?
def chunk_text(text_blocks, chunk_size=512, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", ":", ",", " ", ""],
    )
    joined = "\n\n".join(text_blocks)
    return splitter.split_text(joined)


# Embed and index
# paraphrase-multilingual-MiniLM-L12-v2

def build_faiss_index(chunks, model_name="intfloat/multilingual-e5-small"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    return index, chunks, model_name

# Save everything
def save_index(index, chunks, embed_model_name, save_path="vector_store"):
    os.makedirs(save_path, exist_ok=True)
    faiss.write_index(index, f"{save_path}/doc_index.faiss")
    with open(f"{save_path}/doc_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(f"{save_path}/embedding_model_name.txt", "w") as f:
        f.write(embed_model_name)

if __name__ == "__main__":
    elements = extract_pdf_elements(pdfpath)
    chunks = chunk_text(elements)
    print(chunks[100])
    index, chunks, model_name = build_faiss_index(chunks)
    save_index(index, chunks, model_name)
