import pandas as pd
import numpy as np
import os
import config


def preprocess_csv_dataset():
    """
    Reads the raw CSV, cleans it, generates Power labels (PowerGen),
    and saves a clean dataset ready for training.
    """
    print("[-] Starting Data Preprocessing for CSV Dataset...")

    # 1. Load the Dataset
    try:
        df = pd.read_csv(config.DATASET_PATH)
        print("Column names found in CSV:", list(df.columns))
        print(f"[+] Raw data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"[!] Error loading file: {e}")
        return None

    # 2. Data Cleaning
    # Remove any rows where critical resource columns are missing
    critical_columns = ['CPU_Usage', 'Memory_Usage', 'Network_IO']
    df = df.dropna(subset=critical_columns)

    # Optional: Remove Anomalies (where Anomaly_Label == 1) to ensure the model learns normal patterns
    if 'Anomaly_Label' in df.columns:
        original_count = len(df)
        df = df[df['Anomaly_Label'] == 0]
        removed_count = original_count - len(df)
        print(f"[-] Removed {removed_count} anomalous records for cleaner training.")

    # 3. Feature Engineering (Handling Categories)
    # Convert 'Workload' (e.g., 'Web', 'Database') into numbers using One-Hot Encoding
    if 'Workload_Type' in df.columns:
        print("[-] Converting 'Workload_Type' to numeric features...")
        df = pd.get_dummies(df, columns=['Workload_Type'], prefix='WL')
        print("[+] Workload types converted successfully.")

    # Also, drop 'User_ID' and 'Timestamp' as they are not needed for power prediction
    # and 'User_ID' might be a string which crashes the model
    cols_to_drop = ['User_ID', 'Timestamp', 'Anomaly_Label']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
            print(f"[-] Dropped column: {col}")

    # 4. Generate Power Consumption Target (PowerGen)
    # Formula: Base + (CPU * 50) + (Memory * 30) + (Disk * 20) + Noise
    print("[-] Generating Synthetic Power Labels...")

    # Ensure we only use numerical columns for calculation
    # Select all columns that look like resource usage
    resource_cols = [col for col in df.columns if
                     col in ['CPU_Usage', 'Memory_Usage', 'Disk_IO', 'Network_IO'] or col.startswith('WL_')]

    # Calculate Load Score (Sum of relevant metrics)
    # Note: One-hot encoded columns are 0 or 1, so they add slight overhead
    total_load = df['CPU_Usage'] + df['Memory_Usage'] + df['Disk_IO'] + df['Network_IO']

    # Apply Power Formula
    power_consumption = 100.0 + (total_load * 40.0)
    noise = np.random.normal(0, 5, size=len(df))  # Add randomness
    df['Power_Consumption'] = power_consumption + noise

    # 5. Save Cleaned Data
    # Save to the results folder
    output_path = os.path.join(config.RESULTS_DIR, 'cleaned_cloud_data.csv')
    df.to_csv(output_path)
    print(f"[+] Cleaned dataset saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    # Run preprocessing directly if needed
    preprocess_csv_dataset()