import faiss

from sentence_transformers import SentenceTransformer

 

def create_faiss_index(documents):

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embedder.encode(documents)

    dimension = embeddings.shape[1]

 

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embedder

 

def search_faiss_index(query, index, embedder, documents):

    query_embedding = embedder.encode([query])

    distances, indices = index.search(query_embedding, k=1)

    return documents[indices[0][0]]

 

# Exemple

if __name__ == "__main__":

    docs = ["Afreetech fournit des services technologiques.", "Nous travaillons dans l'innovation."]

    index, embedder = create_faiss_index(docs)

    result = search_faiss_index("Quels sont les services ?", index, embedder, docs)

    print("Résultat :", result)