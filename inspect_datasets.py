from datasets import load_dataset
import pandas as pd

datasets_to_check = [
    "fibonacciai/Digikala-Comments",
    "ParsiAI/digikala-sentiment-analysis"
]

for ds_name in datasets_to_check:
    print(f"\nChecking {ds_name}...")
    try:
        ds = load_dataset(ds_name, split='train')
        df = pd.DataFrame(ds.select(range(min(5, len(ds)))))
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(2))
    except Exception as e:
        print(f"Error: {e}")
