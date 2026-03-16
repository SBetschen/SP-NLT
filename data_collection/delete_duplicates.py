import os
import hashlib

folder = "data"  

hashes = {}
duplicates = []

def file_hash(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)

    if os.path.isfile(filepath):
        h = file_hash(filepath)

        if h in hashes:
            duplicates.append(filepath)
        else:
            hashes[h] = filepath

# Delete duplicates
for dup in duplicates:
    os.remove(dup)
    print(f"Deleted duplicate: {dup}")

print(f"\nTotal duplicates removed: {len(duplicates)}")