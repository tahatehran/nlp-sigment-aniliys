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
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_samples_from_stream(ds_name, split, token, target_count, buffer_size=2000):
    print(f"Streaming samples from {ds_name}...")
    try:
        # EhsanShahbazi datasets might have different structures, we try to handle them
        ds = load_dataset(ds_name, split=split, token=token if token else None, streaming=True)
        samples = []
        count = 0
        for item in ds:
            # Common column names for text in these datasets
            potential_cols = ['text', 'Text', 'comment', 'Comment', 'content', 'Content', 'body']
            text_val = None
            for col in potential_cols:
                if col in item:
                    text_val = item[col]
                    break

            if text_val:
                cleaned = clean_text(text_val)
                # Filter for meaningful length
                if len(cleaned) > 25:
                    samples.append({'text': cleaned})
                    count += 1

            # Stop if we have enough samples or reached buffer limit
            if count >= buffer_size or count >= target_count * 3:
                break

        if not samples:
            return None

        df = pd.DataFrame(samples)
        return df
    except Exception as e:
        print(f"Error loading {ds_name}: {e}")
        return None

def fetch_all_data(target_total=2000):
    """Fetches data from multiple Digikala comment datasets on Hugging Face."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token == "": hf_token = None
    datasets_to_load = [
        ("fibonacciai/Digikala-Comments", "train"),
        ("ParsiAI/digikala-sentiment-analysis", "train"),
        ("EhsanShahbazi/digikala-comments", "train"),
        ("EhsanShahbazi/digikala-comments-with-media", "train")
    ]

    dfs = []
    per_source_target = target_total // len(datasets_to_load)

    for ds_name, split in datasets_to_load:
        df = get_samples_from_stream(ds_name, split, hf_token, per_source_target)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("Warning: No datasets could be loaded. Using mock data for safety.")
        return pd.DataFrame({'text': ["این یک نظر تستی است برای زمانی که دیتاست در دسترس نیست."] * 10})

    processed_dfs = []
    # Aim for balanced sampling
    per_source = target_total // len(dfs)

    for source_df in dfs:
        source_df = source_df.drop_duplicates(subset=['text'])
        if not source_df.empty:
            s_size = min(len(source_df), per_source)
            processed_dfs.append(source_df.sample(n=s_size, random_state=42))

    if not processed_dfs:
        return None

    df_sample = pd.concat(processed_dfs, ignore_index=True)
    # Shuffle the final dataset
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total samples collected: {len(df_sample)}")
    return df_sample

def generate_faiss_index(df, output_dir="data"):
    print("Generating FAISS index...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    texts = df['text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    print(f"FAISS index saved to {output_dir}/faiss_index.bin")

def prepare_data():
    df = fetch_all_data()
    if df is not None:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/digikala_samples.csv", index=False)
        print(f"Data saved to data/digikala_samples.csv")
        generate_faiss_index(df)

if __name__ == "__main__":
    try:
        prepare_data()
        print("Data preparation completed successfully.")
        os._exit(0)
    except Exception as e:
        print(f"Fatal error during data preparation: {e}")
        os._exit(1)
