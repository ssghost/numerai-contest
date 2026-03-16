import numpy as np
import pandas as pd
import json
import os
from constants import RESULT_PATH, DATA_PATH, FEAT_CUST

def merge_data(submission_path, live_data_path, feature_list) -> pd.DataFrame:
    sub_df = pd.read_csv(submission_path)
    live_df = pd.read_parquet(live_data_path, columns=feature_list)
    merged_df = live_df.merge(sub_df, left_index=True, right_on="id", how="inner")
    
    print(f"Merged data length: {len(merged_df)}.")
    return merged_df

def neutralize(df, preds_col, feature_cols, proportion=1.0) -> pd.Series:
    scores = df[preds_col].values
    feats = df[feature_cols].values
    exposures = np.hstack((feats, np.array([np.mean(scores)] * len(feats)).reshape(-1, 1)))
    beta = np.linalg.lstsq(exposures, scores, rcond=None)[0]
    projection = exposures.dot(beta)
    neutralized_scores = scores - (proportion * projection)
    return pd.Series(neutralized_scores).rank(pct=True).values

def run_pipeline(result=RESULT_PATH, live=DATA_PATH, feat=FEAT_CUST) -> None:
    if os.path.exists(result) and os.path.exists(live):
        with open(feat, 'r') as f:
            features = json.load(f)['feature_sets']['custom_features']
        m_data = merge_data(result, live, features)
        
        m_data["prediction_neutral"] = neutralize(m_data, "prediction", features, proportion=0.5)
        final_submission = m_data[["id", "prediction_neutral"]].rename(columns={"prediction_neutral": "prediction"})
        final_submission.to_csv("data/submission_neutralized.csv", index=False)
        print(f"Neutralized result generated with range: {final_submission['prediction'].min():.2f} - {final_submission['prediction'].max():.2f}")
    else:
        print("Error occured.")

if __name__ == "__main__":
    run_pipeline()