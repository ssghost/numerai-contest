import os
from login import login
from constants import DATA_PATH, TRAIN_DATA

def download_live() -> None:
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        napi = login()
        napi.download_dataset("v5.2/live.parquet", DATA_PATH)
        print(f"Data saved at: {DATA_PATH}.")
    except Exception as e:
        print(f"Error occurred: {e}.")

def download_train() -> None:
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        napi = login()
        for ds in TRAIN_DATA:
            target_path = f"data/{ds.split('/')[-1]}"
            if not os.path.exists(target_path):
                napi.download_dataset(ds, target_path)
            else:
                print(f"{ds} already downloaded.")
    except Exception as e:
        print(f"Error occurred: {e}.") 

if __name__ == '__main__':
    download_train()
