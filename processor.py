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

        # 1. Sentiment Model (mBERT)
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1 if self.device == "cpu" else 0
        )

        # 2. Embedding Model for RAG (MiniLM)
        self.embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # 3. GPT2 for Reasoning (Persian optimized GPT2)
        # Using HooshvareLab/gpt2-fa-comment as it is a specialized GPT2 for Persian comments
        self.gen_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa-comment")
        self.gen_model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-fa-comment").to(self.device)

        # Load Data and Build Index
        if not pd.io.common.file_exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found. Please run prepare_data.py first.")

        self.df = pd.read_csv(data_path)
        self.texts = self.df['text'].tolist()

        print("Building FAISS index...")
        embeddings = self.embed_model.encode(self.texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def get_sentiment(self, text):
        result = self.sentiment_pipe(text[:512])[0]
        score = int(result['label'].split()[0])
        return score, result['score']

    def retrieve_similar(self, text, k=2):
        query_vec = self.embed_model.encode([text])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        return [self.texts[i] for i in indices[0]]

    def generate_explanation(self, text, sentiment_score):
        similar_comments = self.retrieve_similar(text)
        # Construct context from similar comments
        context = " ".join([f"نظر مشابه: {c[:80]}" for c in similar_comments])

        sentiment_label = "مثبت" if sentiment_score > 3 else "منفی" if sentiment_score < 3 else "خنثی"

        # Prompt engineering for better reasoning
        prompt = f"متن: {text}\nاحساس: {sentiment_label}\nشواهد: {context}\nدلیل فنی هوش مصنوعی:"

        inputs = self.gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(self.device)

        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_k=40,
            top_p=0.92,
            temperature=0.8,
            pad_token_id=self.gen_tokenizer.eos_token_id
        )

        full_text = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "دلیل فنی هوش مصنوعی:" in full_text:
            explanation = full_text.split("دلیل فنی هوش مصنوعی:")[-1].strip()
        else:
            explanation = "تحلیل بر اساس الگوهای مشابه در دیتاست و کلمات کلیدی موجود در متن کاربر انجام شده است."

        return explanation if len(explanation) > 5 else "با توجه به کلمات استفاده شده و شباهت با نظرات دیگر، این نظر دارای بار احساسی مشخص شده است."

if __name__ == "__main__":
    rag = SentimentRAG()
    test_text = "کیفیتش خوبه ولی قیمتش بالاست"
    score, conf = rag.get_sentiment(test_text)
    print(f"Sentiment: {score}, Confidence: {conf}")
    print(f"Reason: {rag.generate_explanation(test_text, score)}")
