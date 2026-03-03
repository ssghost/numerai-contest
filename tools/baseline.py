import pandas as pd
import numpy as np
from constants import DATA_PATH, RESULT_PATH

def baseline() -> None:
    live_data = pd.read_parquet(DATA_PATH)
    submission = live_data.reset_index()[['id']].copy()
    submission['prediction'] = np.random.uniform(0.45, 0.55, size=len(submission))
    submission.to_csv(RESULT_PATH, index=False)

    print(f"Result file saved at: {RESULT_PATH}.")
    print(f"Total length: {len(submission)}.")

if __name__ == '__main__':
    baseline()