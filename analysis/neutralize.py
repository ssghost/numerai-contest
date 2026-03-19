import numpy as np
import pandas as pd
import json
import os
from constants import RESULT_PATH, DATA_PATH, FEAT_CUST

VAL_DATA_PATH = "data/validation.parquet"
VAL_RESULT_PATH = "data/val_result.csv" 
TARGET_COL = "target_cyrusd_20"

def merge_data(submission_path, live_data_path, feature_list) -> pd.DataFrame:
    sub_df = pd.read_csv(submission_path)
    avai_cols = feature_list + (["era", TARGET_COL] if "validation" in live_data_path else [])
    live_df = pd.read_parquet(live_data_path, columns=avai_cols).astype({
        f: 'float32' for f in feature_list
    })
    merged_df = live_df.merge(sub_df, left_index=True, right_on="id", how="inner")
    
    print(f"Merged data length: {len(merged_df)}.")
    return merged_df

def calc_vector(df, preds_col, feature_cols, proportion=1.0) -> pd.Series:
    scores = df[preds_col].values
    feats = df[feature_cols].values
    exposures = np.hstack((feats, np.ones((len(feats), 1), dtype=np.float32)))
    beta = np.linalg.lstsq(exposures, scores, rcond=None)[0]
    return exposures.dot(beta)

def find_optp(val_result_path, val_data_path, feature_list) -> float:
    if not (os.path.exists(val_result_path) and os.path.exists(val_data_path)):
        return 0.5

    val_data = merge_data(val_result_path, val_data_path, feature_list)
    val_data["projection"] = val_data.groupby("era", group_keys=False).apply(
        lambda x: pd.Series(calc_vector(x, "prediction", feature_list), index=x.index)
    )
    
    proportions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    opt_p = 0.5
    max_sharpe = -np.inf

    for p in proportions:
        neutralized_scores = val_data["prediction"] - (p * val_data["projection"])
        val_data["temp_preds"] = pd.Series(neutralized_scores).rank(pct=True).values
        
        era_scores = val_data.groupby("era").apply(
            lambda x: x["temp_preds"].corr(x[TARGET_COL], method="spearman")
        )
        
        mean_corr = era_scores.mean()
        std_corr = era_scores.std()
        sharpe = mean_corr / std_corr if std_corr > 0 else 0
        print(f"Proportion:{p:10.1f}, M-Corr:{mean_corr:9.4f}, Sharpe:{sharpe:6.4f}.")
        
        if sharpe > max_sharpe:
            max_sharpe = sharpe
            opt_p = p
    
    del val_data["projection"]
    del val_data["temp_preds"]

    print(f"Found opt_p:{opt_p}.")
    return opt_p

def run_pipeline(result=RESULT_PATH, live=DATA_PATH, feat=FEAT_CUST) -> None:
    if os.path.exists(result) and os.path.exists(live):
        with open(feat, 'r') as f:
            features = json.load(f)['feature_sets']['custom_features']
        
        opt_p = find_optp(VAL_RESULT_PATH, VAL_DATA_PATH, features)
        m_data = merge_data(result, live, features)
        scores = m_data["prediction"].values
        proj = calc_vector(m_data, "prediction", features)
        m_data["prediction_neutral"] = pd.Series(scores - opt_p * proj).rank(pct=True).values
        final_submission = m_data[["id", "prediction_neutral"]].rename(columns={"prediction_neutral": "prediction"})
        final_submission.to_csv("data/submission_neutralized.csv", index=False)
        print(f"Neutralized result generated with range: {final_submission['prediction'].min():.2f} - {final_submission['prediction'].max():.2f}")
    else:
        print("Error occured.")

if __name__ == "__main__":
    run_pipeline()