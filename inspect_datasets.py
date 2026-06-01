import sys
import os
from datasets import load_dataset

# Using os._exit to prevent potential thread state issues during CI/CD cleanup
# which can cause exit code 134/139.

datasets_to_check = [
    "fibonacciai/Digikala-Comments",
    "ParsiAI/digikala-sentiment-analysis"
]

def validate_datasets():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    failed = False

    for ds_name in datasets_to_check:
        print(f"Checking accessibility: {ds_name}...")
        try:
            # Streaming + taking 1 item to ensure real connectivity
            ds = load_dataset(ds_name, split='train', token=hf_token, streaming=True)
            _ = next(iter(ds))
            print(f"✅ {ds_name} is accessible.")
        except Exception as e:
            print(f"❌ Error loading {ds_name}: {e}")
            failed = True

    if failed:
        print("\n[!] One or more critical datasets are unavailable.")
        sys.stdout.flush()
        os._exit(1)

    print("\n[+] All upstream datasets are healthy.")
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    validate_datasets()
