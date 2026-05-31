import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import os

class SentimentRAG:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SentimentRAG, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_path="data/digikala_samples.csv"):
        if hasattr(self, 'initialized') and self.initialized:
            return

        print("Initializing SentimentRAG models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HUGGINGFACE_TOKEN")

        # 1. Sentiment Model
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1 if self.device == "cpu" else 0,
            token=hf_token
        )

        # 2. Embedding Model
        self.embed_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            use_auth_token=hf_token
        )

        # 3. Reasoning Model
        self.gen_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa-comment", token=hf_token)
        self.gen_model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-fa-comment", token=hf_token).to(self.device)

        # Load Data
        if not os.path.exists(data_path):
            from prepare_data import prepare_data
            prepare_data()

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} could not be created.")

        self.df = pd.read_csv(data_path)
        self.texts = self.df['text'].tolist()

        print(f"Building FAISS index for {len(self.texts)} items...")
        embeddings = self.embed_model.encode(self.texts, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

        self.initialized = True

    def get_sentiment(self, text):
        # Truncate text to fit model max length
        result = self.sentiment_pipe(text[:512])[0]
        score = int(result['label'].split()[0])
        return score, result['score']

    def retrieve_similar(self, text, k=2):
        query_vec = self.embed_model.encode([text])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        return [self.texts[i] for i in indices[0]]

    def generate_explanation(self, text, sentiment_score):
        similar_comments = self.retrieve_similar(text, k=2)
        context = " ".join([f"نمونه: {c[:60]}" for c in similar_comments])
        sentiment_label = "مثبت" if sentiment_score > 3 else "منفی" if sentiment_score < 3 else "خنثی"

        prompt = f"متن: {text[:100]}\nاحساس: {sentiment_label}\nشواهد: {context}\nدلیل فنی:"
        inputs = self.gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(self.device)

        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.gen_tokenizer.eos_token_id
        )

        full_text = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "دلیل فنی:" in full_text:
            explanation = full_text.split("دلیل فنی:")[-1].strip()
        else:
            explanation = "تحلیل بر اساس الگوهای متنی مشابه در پایگاه داده دیجی‌کالا انجام شده است."

        return explanation if len(explanation) > 10 else "این نظر به دلیل شباهت با نظرات ثبت شده قبلی و الگوهای کلامی شناسایی شده، دارای بار احساسی مشخص شده است."

if __name__ == "__main__":
    rag = SentimentRAG()
    test_text = "کیفیتش خوبه ولی قیمتش بالاست"
    score, conf = rag.get_sentiment(test_text)
    print(f"Sentiment: {score}, Confidence: {conf}")
    print(f"Reason: {rag.generate_explanation(test_text, score)}")
