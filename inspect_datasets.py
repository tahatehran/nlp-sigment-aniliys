import sys
from datasets import load_dataset
import os

datasets_to_check = [
    "fibonacciai/Digikala-Comments",
    "ParsiAI/digikala-sentiment-analysis"
]

def validate_datasets():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    failed = False

    for ds_name in datasets_to_check:
        print(f"Checking {ds_name}...")
        try:
            # Removed trust_remote_code as suggested by warning
            load_dataset(ds_name, split='train', token=hf_token, streaming=True)
            print(f"✅ {ds_name} is accessible.")
        except Exception as e:
            print(f"❌ Error loading {ds_name}: {e}")
            failed = True

    if failed:
        print("\n[!] One or more critical datasets are unavailable. Build failed.")
        sys.exit(1)

    print("\n[+] All upstream datasets are healthy.")
    sys.exit(0)

if __name__ == "__main__":
    validate_datasets()
