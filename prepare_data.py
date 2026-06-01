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
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    print(f"Loading datasets from Hugging Face in streaming mode... (Using token: {'Yes' if hf_token else 'No'})")
    dfs = []

    # We take a buffer of rows to ensure we have enough good samples after filtering.
    # 5000 rows is usually plenty for our 800 target and fits easily in RAM.
    STREAM_BUFFER = 5000

    # 1. fibonacciai/Digikala-Comments
    try:
        ds1 = load_dataset("fibonacciai/Digikala-Comments", split='train', token=hf_token, streaming=True)
        # Convert stream to list of dicts then to DataFrame
        data1 = list(ds1.take(STREAM_BUFFER))
        df1 = pd.DataFrame(data1)
        col = 'Text' if 'Text' in df1.columns else ('comment' if 'comment' in df1.columns else 'text')
        if col in df1.columns:
            df1 = df1.rename(columns={col: 'text'})
            dfs.append(df1[['text']])
            print(f"Loaded {len(df1)} rows from fibonacciai/Digikala-Comments (streaming)")
    except Exception as e:
        print(f"Could not load ds1: {e}")

    # 2. ParsiAI/digikala-sentiment-analysis
    try:
        ds2 = load_dataset("ParsiAI/digikala-sentiment-analysis", split='train', token=hf_token, streaming=True)
        data2 = list(ds2.take(STREAM_BUFFER))
        df2 = pd.DataFrame(data2)
        col = 'Text' if 'Text' in df2.columns else ('comment' if 'comment' in df2.columns else 'text')
        if col in df2.columns:
            df2 = df2.rename(columns={col: 'text'})
            dfs.append(df2[['text']])
            print(f"Loaded {len(df2)} rows from ParsiAI/digikala-sentiment-analysis (streaming)")
    except Exception as e:
        print(f"Could not load ds2: {e}")

    # 3. EhsanShahbazi/digikala-comments (Gated)
    try:
        ds3 = load_dataset("EhsanShahbazi/digikala-comments", split='train', token=hf_token, streaming=True)
        data3 = list(ds3.take(STREAM_BUFFER))
        df3 = pd.DataFrame(data3)
        potential_cols = ['Text', 'comment', 'text', 'Comment']
        found = False
        for col in potential_cols:
            if col in df3.columns:
                df3 = df3.rename(columns={col: 'text'})
                dfs.append(df3[['text']])
                print(f"Loaded {len(df3)} rows from EhsanShahbazi/digikala-comments (streaming)")
                found = True
                break
    except Exception as e:
        print(f"Could not load ds3 (EhsanShahbazi): {e}")

    if not dfs:
        raise RuntimeError("هیچ داده‌ای از Hugging Face بارگذاری نشد. لطفاً اتصال اینترنت یا توکن API خود را بررسی کنید.")

    print("Merging and cleaning...")

    # Process each source to ensure we get a balanced sample if possible
    processed_dfs = []
    target_total = 800
    per_source = target_total // len(dfs)

    for source_df in dfs:
        source_df = source_df.dropna(subset=['text'])
        source_df['text'] = source_df['text'].apply(clean_text)
        source_df = source_df[source_df['text'].str.len() > 25]
        source_df = source_df.drop_duplicates(subset=['text'])

        if not source_df.empty:
            s_size = min(len(source_df), per_source)
            processed_dfs.append(source_df.sample(n=s_size, random_state=42))

    if not processed_dfs:
        raise RuntimeError("پس از فیلتر کردن و پاکسازی، هیچ داده معتبری باقی نماند.")

    df_sample = pd.concat(processed_dfs, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    output_path = "data/digikala_samples.csv"
    df_sample.to_csv(output_path, index=False)

    if not os.path.exists(output_path):
        raise RuntimeError(f"خطا در ایجاد فایل نهایی در مسیر {output_path}")

    print(f"Successfully saved {len(df_sample)} samples to {output_path}")

if __name__ == "__main__":
    try:
        prepare_data()
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
