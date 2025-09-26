import os
import pandas as pd

# Paths
csv_path = r"data/Sri_Lankan_Dataset/Imagewise_Data.csv"
img_dir = r"data/Sri_Lankan_Dataset/Images"

# Load CSV
df = pd.read_csv(csv_path)

# Clean image_name column: strip whitespace, ensure .jpg extension
fixed_names = []
for name in df['image_name']:
    name = str(name).strip()
    if not name.lower().endswith('.jpg'):
        name = name + '.jpg'
    fixed_names.append(name)
df['image_name'] = fixed_names

# Print first 10 and check if they exist
print("First 10 filenames and existence:")
exists_mask = []
for name in df['image_name'][:10]:
    exists = os.path.isfile(os.path.join(img_dir, name))
    print(f"{name}: {'FOUND' if exists else 'MISSING'}")

# Remove rows with missing images
full_exists_mask = [os.path.isfile(os.path.join(img_dir, name)) for name in df['image_name']]
num_missing = len(df) - sum(full_exists_mask)
df_clean = df[full_exists_mask].reset_index(drop=True)

# Save fixed CSV (backup original first)
backup_path = csv_path + ".bak"
if not os.path.exists(backup_path):
    os.rename(csv_path, backup_path)
df_clean.to_csv(csv_path, index=False)
print(f"\nRemoved {num_missing} rows with missing images.")
print(f"CSV cleaned and saved. Original backed up as {backup_path}")
