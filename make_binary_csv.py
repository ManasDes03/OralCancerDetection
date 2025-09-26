import pandas as pd
import os

csv_path = r"data/Sri_Lankan_Dataset/Imagewise_Data.csv"
img_dir = r"data/Sri_Lankan_Dataset/Images"

df = pd.read_csv(csv_path)

# Print unique values and counts in Category column
print("Unique values in 'Category' and their counts:")
print(df['Category'].value_counts())

# Define which values mean 'cancer' and 'non-cancer'
# (You may need to adjust these based on your dataset's actual values)
cancer_labels = ["OCA", "Oral Cancer", 1, 3]  # Example: OCA, Oral Cancer, 1, 3
non_cancer_labels = ["Healthy", "Benign", "OPMD", 0, 2]  # Example: Healthy, Benign, OPMD, 0, 2

def map_label(val):
    if val in cancer_labels:
        return 1
    elif val in non_cancer_labels:
        return 0
    else:
        return None

# Map and filter
df['binary_label'] = df['Category'].apply(map_label)
df_clean = df[df['binary_label'].isin([0, 1])].copy()
print(f"\nRows kept for binary classification: {len(df_clean)} / {len(df)}")

# Overwrite Category column with binary label for training
# (or change your data loader to use 'binary_label' instead)
df_clean['Category'] = df_clean['binary_label']
df_clean = df_clean.drop(columns=['binary_label'])

# Save cleaned CSV (backup original first)
backup_path = csv_path + ".bak2"
if not os.path.exists(backup_path):
    os.rename(csv_path, backup_path)
df_clean.to_csv(csv_path, index=False)
print(f"\nCleaned CSV saved for binary classification. Original backed up as {backup_path}")
