from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger le modèle fine-tuné
model_dir = "./fine_tuned_llama"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

# Fonction pour interagir avec le modèle
def chat_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Tester une question
if __name__ == "__main__":
    while True:
        user_input = input("Vous : ")
        response = chat_with_model(user_input)
        print("Chatbot :", response)
