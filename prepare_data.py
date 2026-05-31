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
    # Use token if available in environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    print(f"Loading datasets from Hugging Face... (Using token: {'Yes' if hf_token else 'No'})")
    dfs = []

    # 1. fibonacciai/Digikala-Comments
    try:
        ds1 = load_dataset("fibonacciai/Digikala-Comments", split='train', token=hf_token)
        df1 = pd.DataFrame(ds1)
        col = 'Text' if 'Text' in df1.columns else ('comment' if 'comment' in df1.columns else 'text')
        if col in df1.columns:
            df1 = df1.rename(columns={col: 'text'})
            dfs.append(df1[['text']])
            print(f"Loaded {len(df1)} rows from fibonacciai/Digikala-Comments")
    except Exception as e:
        print(f"Could not load ds1: {e}")

    # 2. ParsiAI/digikala-sentiment-analysis
    try:
        ds2 = load_dataset("ParsiAI/digikala-sentiment-analysis", split='train', token=hf_token)
        df2 = pd.DataFrame(ds2)
        col = 'Text' if 'Text' in df2.columns else ('comment' if 'comment' in df2.columns else 'text')
        if col in df2.columns:
            df2 = df2.rename(columns={col: 'text'})
            dfs.append(df2[['text']])
            print(f"Loaded {len(df2)} rows from ParsiAI/digikala-sentiment-analysis")
    except Exception as e:
        print(f"Could not load ds2: {e}")

    # 3. EhsanShahbazi/digikala-comments (Gated)
    try:
        ds3 = load_dataset("EhsanShahbazi/digikala-comments", split='train', token=hf_token)
        df3 = pd.DataFrame(ds3)
        # Check potential column names for this specific dataset
        potential_cols = ['Text', 'comment', 'text', 'Comment']
        found = False
        for col in potential_cols:
            if col in df3.columns:
                df3 = df3.rename(columns={col: 'text'})
                dfs.append(df3[['text']])
                print(f"Loaded {len(df3)} rows from EhsanShahbazi/digikala-comments using col '{col}'")
                found = True
                break
        if not found:
            print(f"Columns in ds3: {df3.columns.tolist()}")
    except Exception as e:
        print(f"Could not load ds3 (EhsanShahbazi): {e}")

    if not dfs:
        print("ERROR: No data could be loaded.")
        return

    print("Merging and cleaning...")
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 25]
    df = df.drop_duplicates(subset=['text'])

    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    os.makedirs("data", exist_ok=True)
    df_sample.to_csv("data/digikala_samples.csv", index=False)
    print(f"Successfully saved {len(df_sample)} samples to data/digikala_samples.csv")

if __name__ == "__main__":
    prepare_data()
