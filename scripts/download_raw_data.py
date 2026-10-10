import os
import zipfile
import urllib.request

RAW_DIR = os.path.join("data", "raw")
LIAR_DIR = os.path.join(RAW_DIR, "liar")
ZIP_PATH = os.path.join(RAW_DIR, "liar_dataset.zip")
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

os.makedirs(LIAR_DIR, exist_ok=True)

print(f"Downloading LIAR dataset from {LIAR_URL}...")
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(LIAR_URL, headers=headers)

with urllib.request.urlopen(req) as response, open(ZIP_PATH, 'wb') as out_file:
    out_file.write(response.read())

print("Extracting files...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(LIAR_DIR)

# Cleanup the zip file
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

print(f"Dataset successfully placed in: {os.path.abspath(LIAR_DIR)}")
print("Files extracted:", os.listdir(LIAR_DIR))