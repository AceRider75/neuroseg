import os
import sys
from src.data.dataset import get_data_split

# Add project root to path
sys.path.append(os.getcwd())

def list_test_samples():
    print("--- NeuroSeg Test Data Finder ---")
    print("This script identifies patients that were EXCLUDED from the training set.\n")
    
    try:
        _, _, test_df = get_data_split("archive")
    except Exception as e:
        print(f"Error loading splits: {e}")
        return

    test_patients = test_df['patient_id'].unique()
    
    print(f"Found {len(test_patients)} patients in the held-out TEST SET.")
    print("These patients were never seen by the model if you ran train.py.\n")
    
    print("Example patient folders for testing:")
    for patient in test_patients[:10]:
        # Count slices for this patient
        num_slices = len(test_df[test_df['patient_id'] == patient])
        print(f"- {patient} ({num_slices} slices)")
        
    print("\nPick any slice from these folders to test the model's generalization!")
    print("Path format: archive/kaggle_3m/<FOLDER_NAME>/<IMAGE_NAME>.tif")

if __name__ == "__main__":
    list_test_samples()
