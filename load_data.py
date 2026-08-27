import pandas as pd

# Load LIAR (LIAR is typically formatted as a TSV - Tab Separated Values)
# Update the filename to match whatever you named your local file
liar_data = pd.read_csv('data/train.tsv', sep='\t', header=None)
print(f"LIAR dataset loaded with {len(liar_data)} rows.")

# Load FEVER (FEVER is typically formatted as JSONL - JSON Lines)
fever_data = pd.read_json('data/train.jsonl', lines=True)
print(f"FEVER dataset loaded with {len(fever_data)} rows.")

# Remember to slice this down to a 10,000 claim subsample 
# later so it fits within your 8GB VRAM limit!
fever_subsample = fever_data.head(10000)