import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

class SentimentRAG:
    def __init__(self, data_path="data/digikala_samples.csv"):
        print("Initializing models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Sentiment Model
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1 if self.device == "cpu" else 0
        )

        # 2. Embedding Model for RAG
        self.embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # 3. GPT2 for Reasoning
        self.gen_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa-comment")
        self.gen_model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-fa-comment").to(self.device)

        # Load Data and Build Index
        self.df = pd.read_csv(data_path)
        self.texts = self.df['text'].tolist()

        print("Building FAISS index...")
        embeddings = self.embed_model.encode(self.texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def get_sentiment(self, text):
        result = self.sentiment_pipe(text[:512])[0]
        # Label is "1 star", "2 stars", etc.
        score = int(result['label'].split()[0])
        return score, result['score']

    def retrieve_similar(self, text, k=2):
        query_vec = self.embed_model.encode([text])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        return [self.texts[i] for i in indices[0]]

    def generate_explanation(self, text, sentiment_score):
        similar_comments = self.retrieve_similar(text)
        context = "\n".join([f"- {c[:100]}..." for c in similar_comments])

        sentiment_label = "مثبت" if sentiment_score > 3 else "منفی" if sentiment_score < 3 else "خنثی"

        prompt = f"نظر کاربر: {text}\nاحساس تشخیص داده شده: {sentiment_label}\nنظرات مشابه:\n{context}\nدلیل هوش مصنوعی برای این تحلیل:"

        # Limit prompt length for GPT2
        inputs = self.gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.gen_tokenizer.eos_token_id
        )

        full_text = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        explanation = full_text.split("دلیل هوش مصنوعی برای این تحلیل:")[-1].strip()
        return explanation

# Quick test if run directly
if __name__ == "__main__":
    rag = SentimentRAG()
    test_text = "واقعا عالی بود، خیلی راضی هستم از کیفیتش"
    score, conf = rag.get_sentiment(test_text)
    print(f"Sentiment: {score} stars (conf: {conf:.2f})")
    explanation = rag.generate_explanation(test_text, score)
    print(f"Explanation: {explanation}")
