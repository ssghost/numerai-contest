import os
import login
from constant import DATA_PATH

def download() -> None:
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        napi = login()
        napi.download_dataset("v5.2/live.parquet", DATA_PATH)
        print(f"Data saved at: {DATA_PATH}.")
    except Exception as e:
        print(f"Error occurred: {e}.")


if __name__ == '__main__':
    download()