import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Load and clean the raw text from input.txt
def load_text():
    texts = []
    with open("input.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("🔹") and not "Sentiment" in line:
                texts.append(line)
    return texts

# Step 2: Fit tokenizer on texts
texts = load_text()
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Step 3: Save tokenizer to tokenizer.pkl
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ tokenizer.pkl file created successfully.")
