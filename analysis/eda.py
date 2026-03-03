import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from constants import FEAT_PATH, TRAIN_PATH

def run_eda_subset() -> None:
    with open(FEAT_PATH, 'r') as f:
        features_config = json.load(f)
    small_features = features_config['feature_sets']['small']
    columns_to_read = ['id', 'era', 'target'] + small_features
    
    if not os.path.exists(TRAIN_PATH):
        print(f"Can't find {TRAIN_PATH}.")
        return

    df = pd.read_parquet(TRAIN_PATH, columns=columns_to_read)

    print(f"\nLoaded data of {len(df):,} lines.")
    print(f"Memory used: {df.memory_usage().sum() / 1e6:.2f} MB")

    for feat in small_features[:5]:
        print(df[feat].value_counts(normalize=True).sort_index())

    era_counts = df.groupby('era').size()
    print(f"Era counts: {era_counts.head()}")

    missing = df.isnull().sum().sum()
    print(f"Missing data: {missing}")

def run_era_corr() -> None:
    sns.set_theme(style="whitegrid")
    with open(FEAT_PATH, 'r') as f:
        features_config = json.load(f)
    small_features = features_config['feature_sets']['small']
    
    df = pd.read_parquet(TRAIN_PATH, columns=['era', 'target'] + small_features)
    era_correlations = df.groupby('era').apply(
        lambda x: x[small_features].corrwith(x['target'], method='spearman')
    )

    mean_corr = era_correlations.mean()
    top_features = mean_corr.abs().sort_values(ascending=False).head(5).index.tolist()
    
    print("\nTop 5 most corr features:")
    for feat in top_features:
        print(f"- {feat}: {mean_corr[feat]:.4f}")

    best_feat = top_features[0]
    plt.figure(figsize=(15, 6))
    plot_data = era_correlations[best_feat].reset_index()
    plot_data.columns = ['era', 'correlation']
    sns.barplot(
        data=plot_data, 
        x='era', 
        y='correlation', 
        palette="vlag" 
    )
    plt.title(f"Feature Stability Over Eras: {best_feat}")
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axhline(y=mean_corr[best_feat], color='r', linestyle='--', label=f'Mean Corr: {mean_corr[best_feat]:.4f}')
    plt.xticks([]) 
    plt.ylabel("Spearman Correlation")
    plt.legend()
    output_img_1 = "img/top_feature_stability.png"
    plt.savefig(output_img_1)
    print(f"\nStability chart saved at: {output_img_1}.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        era_correlations[top_features].corr(), 
        annot=True, 
        cmap='coolwarm', 
        center=0
    )
    plt.title("Correlation Matrix of Top 5 Features")
    output_img_2 = "img/top_features_heatmap.png"
    plt.savefig(output_img_2)
    print(f"\nCorrelation heatmap saved at: {output_img_2}.") 

if __name__ == "__main__":
    run_era_corr()