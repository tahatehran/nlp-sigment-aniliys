import pandas as pd
from datasets import load_dataset
import re
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_samples_from_stream(ds_name, split, token, target_count, buffer_size=1000):
    print(f"Streaming samples from {ds_name}...")
    try:
        ds = load_dataset(ds_name, split=split, token=token, streaming=True)
        samples = []
        count = 0
        for item in ds:
            potential_cols = ['text', 'Text', 'comment', 'Comment']
            text_val = None
            for col in potential_cols:
                if col in item:
                    text_val = item[col]
                    break

            if text_val:
                cleaned = clean_text(text_val)
                if len(cleaned) > 25:
                    samples.append({'text': cleaned})
                    count += 1

            if count >= buffer_size or count >= target_count * 2:
                break

        if not samples:
            return None

        df = pd.DataFrame(samples)
        return df
    except Exception:
        return None

def fetch_all_data(target_total=500):
    """Lighter data fetching for online fallback."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    datasets_to_load = [
        ("fibonacciai/Digikala-Comments", "train"),
        ("ParsiAI/digikala-sentiment-analysis", "train")
    ]

    dfs = []
    for ds_name, split in datasets_to_load:
        df = get_samples_from_stream(ds_name, split, hf_token, target_total // 2)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None

    processed_dfs = []
    per_source = target_total // len(dfs)

    for source_df in dfs:
        source_df = source_df.drop_duplicates(subset=['text'])
        if not source_df.empty:
            s_size = min(len(source_df), per_source)
            processed_dfs.append(source_df.sample(n=s_size, random_state=42))

    if not processed_dfs:
        return None

    df_sample = pd.concat(processed_dfs, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_sample

def generate_faiss_index(df, output_dir="data"):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    texts = df['text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))

def prepare_data():
    df = fetch_all_data()
    if df is not None:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/digikala_samples.csv", index=False)
        generate_faiss_index(df)

if __name__ == "__main__":
    try:
        prepare_data()
        os._exit(0)
    except Exception:
        os._exit(1)
