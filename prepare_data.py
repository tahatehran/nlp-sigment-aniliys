import pandas as pd
from datasets import load_dataset
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data():
    print("Loading datasets...")
    dfs = []

    # 1. fibonacciai/Digikala-Comments
    try:
        ds1 = load_dataset("fibonacciai/Digikala-Comments", split='train')
        df1 = pd.DataFrame(ds1)
        if 'Text' in df1.columns:
            df1 = df1.rename(columns={'Text': 'text', 'Suggestion': 'label'})
            dfs.append(df1[['text']])
            print("Loaded ds1")
    except Exception as e:
        print(f"Error loading ds1: {e}")

    # 2. ParsiAI/digikala-sentiment-analysis
    try:
        ds2 = load_dataset("ParsiAI/digikala-sentiment-analysis", split='train')
        df2 = pd.DataFrame(ds2)
        if 'Text' in df2.columns:
            df2 = df2.rename(columns={'Text': 'text', 'Suggestion': 'label'})
            dfs.append(df2[['text']])
            print("Loaded ds2")
    except Exception as e:
        print(f"Error loading ds2: {e}")

    # 3. EhsanShahbazi/digikala-comments (Gated, likely to fail here, but we have enough from 1 & 2)
    try:
        ds3 = load_dataset("EhsanShahbazi/digikala-comments", split='train')
        df3 = pd.DataFrame(ds3)
        # Assuming typical columns if it were accessible
        for col in ['Text', 'comment', 'text']:
            if col in df3.columns:
                df3 = df3.rename(columns={col: 'text'})
                dfs.append(df3[['text']])
                print("Loaded ds3")
                break
    except Exception as e:
        print(f"Skipping ds3 (gated or error): {e}")

    if not dfs:
        print("No data loaded!")
        return

    print("Merging and cleaning...")
    # Combine all
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 20] # Remove very short comments for better RAG

    # Drop duplicates
    df = df.drop_duplicates(subset=['text'])

    # Select 500 samples for RAG as requested
    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    # Save to CSV
    df_sample.to_csv("data/digikala_samples.csv", index=False)
    print(f"Saved {len(df_sample)} samples to data/digikala_samples.csv")

if __name__ == "__main__":
    prepare_data()
