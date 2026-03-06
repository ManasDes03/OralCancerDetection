"""
prepare_gdc_data.py
-------------------
Process GDC oral cancer dataset from local images and create processed labels for binary classification tasks.
Images are organized in numbered folders (1/, 2/, 3/, ...) corresponding to case numbers.
Handles two classification tasks: Suspicious vs Non-suspicious, and High-risk vs Low-risk.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob


def _norm_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _map_labels_from_row(prov_diagnosis, specialist_diagnosis, risk):
    """
    Map suspicious and risk labels from available columns.
    Priority:
    1) Explicit specialist/risk columns
    2) Provisional diagnosis mapping based on table categories
    """
    prov = _norm_text(prov_diagnosis)
    specialist = _norm_text(specialist_diagnosis)
    risk_norm = _norm_text(risk)

    suspicious_label = None
    risk_label = None

    # Explicit table exclusions (Not applicable in reader-vs-AI table)
    not_applicable_provisional = {
        'oral submucous fibrosis',
        'oral submucosal fibrosis',
        'non-diagnostic procedure finding',
    }
    if prov in not_applicable_provisional:
        return None, None

    # Primary: specialist diagnosis column if present
    if 'non suspicious' in specialist or 'non-suspicious' in specialist:
        suspicious_label = 0
    elif 'suspicious' in specialist:
        suspicious_label = 1

    # Primary: risk column if present
    if 'low risk' in risk_norm or 'low-risk' in risk_norm:
        risk_label = 0
    elif 'high risk' in risk_norm or 'high-risk' in risk_norm:
        risk_label = 1

    # Fallback: provisional diagnosis mapping (from categorization table)
    provisional_map = {
        'oral cavity normal': (0, 0),
        'benign': (0, 0),
        'other': (0, 0),
        'smokeless tobacco keratosis': (1, 0),
        'homogenous oral leukoplakia': (1, 0),
        'homogeneous oral leukoplakia': (1, 0),
        'oral lichen planus': (1, 0),
        'speckled oral leukoplakia': (1, 1),
        'erythroplakia': (1, 1),
        'verrucous oral leukoplakia': (1, 1),
        'proliferative verrucous leukoplakia': (1, 1),
        'proliferative verrucous oral leukoplakia': (1, 1),
        'squamous cell carcinoma of oral mucous membrane': (1, 1),
    }

    if prov in provisional_map:
        prov_suspicious, prov_risk = provisional_map[prov]
        if suspicious_label is None:
            suspicious_label = prov_suspicious
        if risk_label is None:
            risk_label = prov_risk

    # Not-applicable / ambiguous diagnosis
    if suspicious_label is None:
        return None, None

    return suspicious_label, risk_label

def prepare_gdc_dataset(csv_path, images_dir, output_csv_path):
    """
    Prepare GDC dataset by reading local images and creating processed labels.
    
    Args:
        csv_path: Path to original labels.csv
        images_dir: Directory containing images organized in numbered folders
        output_csv_path: Path to save processed labels CSV
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} cases from CSV")
    
    # Prepare processed data
    processed_rows = []
    
    # Track statistics
    total_images = 0
    missing_folders = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
        case_no = row['No.']
        prov_diagnosis = row['Provisional diagnosis']
        specialist_diagnosis = row['Specialist diagnosis']
        risk = row['Risk']
        
        # Skip if specialist diagnosis is NA or missing
        if pd.isna(specialist_diagnosis) or specialist_diagnosis == 'NA':
            continue
        
        # Map to binary labels using specialist/risk columns with fallback
        suspicious_label, risk_label = _map_labels_from_row(
            prov_diagnosis=prov_diagnosis,
            specialist_diagnosis=specialist_diagnosis,
            risk=risk,
        )
        if suspicious_label is None:
            continue
        
        # Look for images in the case folder
        case_folder = os.path.join(images_dir, str(case_no))
        
        if not os.path.exists(case_folder):
            missing_folders.append(case_no)
            continue
        
        # Get all images in the case folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(case_folder, ext)))
        
        if not image_files:
            continue
        
        # Process each image in the folder
        for image_path in image_files:
            image_filename = os.path.basename(image_path)
            relative_path = f"{case_no}/{image_filename}"
            
            # Add row to processed data
            processed_rows.append({
                'case_no': case_no,
                'image_path': relative_path,
                'provisional_diagnosis': prov_diagnosis,
                'specialist_diagnosis': specialist_diagnosis,
                'suspicious_label': suspicious_label,
                'risk': risk,
                'risk_label': risk_label,
                'age': row.get('Age', None),
                'gender': row.get('Gender', None),
            })
            total_images += 1
    
    # Create processed DataFrame
    processed_df = pd.DataFrame(processed_rows)
    
    # Remove duplicate rows (same case/image pair appears multiple times in source sheets)
    before_dedup = len(processed_df)
    processed_df = processed_df.drop_duplicates(subset=['case_no', 'image_path']).copy()
    removed_dups = before_dedup - len(processed_df)

    # Save processed labels
    processed_df.to_csv(output_csv_path, index=False)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total images processed: {len(processed_df)}")
    print(f"Duplicate rows removed: {removed_dups}")
    print(f"Cases processed: {processed_df['case_no'].nunique()}")
    print(f"Suspicious cases: {(processed_df['suspicious_label']==1).sum()}")
    print(f"Non-suspicious cases: {(processed_df['suspicious_label']==0).sum()}")
    print(f"High-risk cases: {(processed_df['risk_label']==1).sum()}")
    print(f"Low-risk cases: {(processed_df['risk_label']==0).sum()}")
    print(f"NA risk cases: {processed_df['risk_label'].isna().sum()}")
    if missing_folders:
        print(f"\nWarning: {len(missing_folders)} case folders not found")
    print(f"\nProcessed labels saved to: {output_csv_path}")
    
    return processed_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GDC dataset for training')
    parser.add_argument('--csv', default='data/gdc_oral_cancer_dataset/labels.csv',
                        help='Path to original labels CSV')
    parser.add_argument('--images-dir', default='data/gdc_oral_cancer_dataset/images',
                        help='Directory containing images in numbered folders')
    parser.add_argument('--output-csv', default='data/gdc_oral_cancer_dataset/processed_labels.csv',
                        help='Path to save processed labels CSV')
    
    args = parser.parse_args()
    
    prepare_gdc_dataset(
        csv_path=args.csv,
        images_dir=args.images_dir,
        output_csv_path=args.output_csv
    )
