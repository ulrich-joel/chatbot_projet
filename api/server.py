from fastapi import FastAPI

from models.load_model import load_model

from knowledge_base.faiss_search import create_faiss_index, search_faiss_index

 

app = FastAPI()

 

# Charger le modèle et la base de connaissances

model, tokenizer = load_model()

documents = ["Afreetech fournit des services technologiques.", "Nous travaillons dans l'innovation."]

index, embedder = create_faiss_index(documents)

 

@app.post("/chat")

async def chat_endpoint(question: str):

    # Recherche dans la base de connaissances

    try:

        response = search_faiss_index(question, index, embedder, documents)

    except:

        # Si pas de réponse, demander au modèle

        inputs = tokenizer(question, return_tensors="pt").to("cuda")

        outputs = model.generate(inputs["input_ids"], max_length=200)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}

 

# Lancez avec `uvicorn api.server:app --reload`