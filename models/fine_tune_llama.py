from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import os
import glob

# Charger le modèle pré-entraîné (LLaMA 3.3-70B)
model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Configuration pour la quantification 8 bits
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Charger le tokenizer et le modèle avec quantification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",  # Automatique selon le matériel disponible
    quantization_config=quantization_config,
)

# Configurer LoRA (fine-tuning efficace)
lora_config = LoraConfig(
    r=16,  # Taille des matrices de rang réduit
    lora_alpha=32,  # Facteur de mise à l'échelle
    target_modules=["q_proj", "v_proj"],  # Modules spécifiques au modèle
    lora_dropout=0.1,  # Taux de dropout
    bias="none",
    task_type="CAUSAL_LM",  # Fine-tuning pour un modèle génératif
)
model = get_peft_model(model, lora_config)

# Chemin des données JSON
models_dir = os.path.join(os.getenv("USERPROFILE"), "Documents", "cyberlab", "chatbot", "models")

# Récupérer tous les fichiers JSON
json_files = glob.glob(os.path.join(models_dir, "*.json"))
if not json_files:
    raise FileNotFoundError(f"Aucun fichier JSON trouvé dans le dossier : {models_dir}")

print("Fichiers JSON trouvés :", json_files)

# Charger les données
dataset = load_dataset("json", data_files=json_files)
print("Dataset chargé :", dataset)

# Préparer les données pour le modèle
def preprocess_function(examples):
    return tokenizer(
        examples["instruction"],
        text_pair=examples["response"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)
print("Dataset tokenisé :", tokenized_dataset)

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",
    per_device_train_batch_size=2,  # Réduire la taille pour s'adapter à la VRAM disponible
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  # Utilisation de précision mixte si compatible
    optim="adamw_bnb_8bit",  # Optimiseur compatible avec 8-bit
)

# Créer un Trainer pour le fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Fine-tuner le modèle
if __name__ == "__main__":
    print("Démarrage du fine-tuning...")
    trainer.train()
    print("Fine-tuning terminé !")

    # Sauvegarder le modèle fine-tuné
    model.save_pretrained("./fine_tuned_llama")
    tokenizer.save_pretrained("./fine_tuned_llama")
    print("Modèle fine-tuné sauvegardé dans ./fine_tuned_llama")
