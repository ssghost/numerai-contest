import pandas as pd
import warnings
import json
import gc

from constants import TRAIN_PATH, FEAT_PATH

warnings.simplefilter(action='ignore', category=RuntimeWarning)

def load_features(path, features) -> pd.DataFrame:
    cols = ['target', 'era'] + features
    df = pd.read_parquet(path, columns=cols, engine='pyarrow')
    df = df.iloc[::4] 
    
    if df['era'].dtype == 'object':
        df['era'] = df['era'].str.replace('era', '').astype(int)
    else:
        df['era'] = df['era'].astype(int)
    
    return df

def calculate_sharpe(df, features) -> pd.DataFrame:
    feature_corrs = []
    
    for era, era_df in df.groupby('era'):
        corr = era_df[features].corrwith(era_df['target'], method='spearman')
        feature_corrs.append(corr)
        
    corr_matrix = pd.concat(feature_corrs, axis=1) 
    mean_corr = corr_matrix.mean(axis=1)
    std_corr = corr_matrix.std(axis=1)
    
    sharpe_ratios = mean_corr / (std_corr + 1e-8)
    results = pd.DataFrame({
        'feature': features,
        'mean_corr': mean_corr.values,
        'std_corr': std_corr.values,
        'sharpe': sharpe_ratios.values
    }).sort_values(by='sharpe', ascending=False)
    
    return results

def main() -> None:
    OUTPUT_JSON = "data/custom_features.json"
    k = 500

    with open(FEAT_PATH, 'r') as f:
        all_features = json.load(f)
        features = all_features['feature_sets']['all']

    df = load_features(TRAIN_PATH, features)
    stability_results = calculate_sharpe(df, features)
    top_k_features = stability_results.head(k)

    custom_set = {
        "feature_sets": {
            "custom_features": top_k_features['feature'].tolist()
        }
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(custom_set, f, indent=4)
    
    print(f"Selected features saved at: {OUTPUT_JSON}")
    del df, stability_results
    gc.collect()

if __name__ == "__main__":
    main()