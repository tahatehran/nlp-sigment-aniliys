import pandas as pd
from datasets import load_dataset
import re
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data():
    print("Loading datasets from Hugging Face...")
    dfs = []

    # 1. fibonacciai/Digikala-Comments
    try:
        ds1 = load_dataset("fibonacciai/Digikala-Comments", split='train')
        df1 = pd.DataFrame(ds1)
        # Identify text column: it's 'Text' based on inspection
        col = 'Text' if 'Text' in df1.columns else ('comment' if 'comment' in df1.columns else 'text')
        if col in df1.columns:
            df1 = df1.rename(columns={col: 'text'})
            dfs.append(df1[['text']])
            print(f"Loaded {len(df1)} rows from fibonacciai/Digikala-Comments")
    except Exception as e:
        print(f"Could not load ds1: {e}")

    # 2. ParsiAI/digikala-sentiment-analysis
    try:
        ds2 = load_dataset("ParsiAI/digikala-sentiment-analysis", split='train')
        df2 = pd.DataFrame(ds2)
        col = 'Text' if 'Text' in df2.columns else ('comment' if 'comment' in df2.columns else 'text')
        if col in df2.columns:
            df2 = df2.rename(columns={col: 'text'})
            dfs.append(df2[['text']])
            print(f"Loaded {len(df2)} rows from ParsiAI/digikala-sentiment-analysis")
    except Exception as e:
        print(f"Could not load ds2: {e}")

    # 3. EhsanShahbazi/digikala-comments
    # Note: This is gated. If no token, it fails. We'll try, but handle gracefully.
    try:
        ds3 = load_dataset("EhsanShahbazi/digikala-comments", split='train')
        df3 = pd.DataFrame(ds3)
        col = 'Text' if 'Text' in df3.columns else ('comment' if 'comment' in df3.columns else 'text')
        if col in df3.columns:
            df3 = df3.rename(columns={col: 'text'})
            dfs.append(df3[['text']])
            print(f"Loaded {len(df3)} rows from EhsanShahbazi/digikala-comments")
    except Exception as e:
        print(f"Note: EhsanShahbazi/digikala-comments skipped (usually requires login/gated access). {e}")

    if not dfs:
        print("ERROR: No data could be loaded from any of the sources.")
        # Create a dummy to avoid crash if necessary, but better to fail early
        return

    print("Merging and cleaning...")
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 25]
    df = df.drop_duplicates(subset=['text'])

    # User requested around 300-500 samples
    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    os.makedirs("data", exist_ok=True)
    df_sample.to_csv("data/digikala_samples.csv", index=False)
    print(f"Successfully saved {len(df_sample)} samples to data/digikala_samples.csv")

if __name__ == "__main__":
    prepare_data()
