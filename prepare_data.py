import pandas as pd
from datasets import load_dataset
import re
import os
import sys

# Increase the robustness of the script against exit crashes
# by using os._exit at the very end if needed.

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_samples_from_stream(ds_name, split, token, target_count, buffer_size=3000):
    print(f"Streaming samples from {ds_name}...")
    try:
        ds = load_dataset(ds_name, split=split, token=token, streaming=True)
        samples = []
        count = 0
        # Iterate manually to have full control and avoid loading too much
        for item in ds:
            # Detect text column
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

            # Stop if we have enough for a good sample base
            if count >= buffer_size:
                break

        if not samples:
            print(f"⚠️ No valid samples found in {ds_name}")
            return None

        df = pd.DataFrame(samples)
        print(f"✅ Extracted {len(df)} samples from {ds_name}")
        return df
    except Exception as e:
        print(f"❌ Could not stream from {ds_name}: {e}")
        return None

def prepare_data():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    print(f"Starting data preparation (Streaming Mode)...")

    datasets_to_load = [
        ("fibonacciai/Digikala-Comments", "train"),
        ("ParsiAI/digikala-sentiment-analysis", "train"),
        ("EhsanShahbazi/digikala-comments", "train")
    ]

    dfs = []
    for ds_name, split in datasets_to_load:
        df = get_samples_from_stream(ds_name, split, hf_token, 400) # Target 400 per source for buffer
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("ERROR: No data could be loaded from any source.")
        sys.stdout.flush()
        os._exit(1)

    print("Merging and final sampling...")

    # Process each source to ensure we get a balanced sample
    processed_dfs = []
    target_total = 800
    per_source = target_total // len(dfs)

    for source_df in dfs:
        source_df = source_df.drop_duplicates(subset=['text'])
        if not source_df.empty:
            s_size = min(len(source_df), per_source)
            processed_dfs.append(source_df.sample(n=s_size, random_state=42))

    if not processed_dfs:
        print("ERROR: No data left after filtering.")
        sys.stdout.flush()
        os._exit(1)

    df_sample = pd.concat(processed_dfs, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    output_path = "data/digikala_samples.csv"
    df_sample.to_csv(output_path, index=False)

    print(f"Successfully saved {len(df_sample)} samples to {output_path}")
    sys.stdout.flush()
    # Force exit to prevent GIL release issues on some environments
    os._exit(0)

if __name__ == "__main__":
    try:
        prepare_data()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.stdout.flush()
        os._exit(1)
