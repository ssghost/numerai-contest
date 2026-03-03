import json
import os
from constants import FEAT_PATH

def view_features() -> None:
    if not os.path.exists(FEAT_PATH):
        pass

    with open(FEAT_PATH, 'r') as f:
        data = json.load(f)

    feature_sets = data.get('feature_sets', {})
    print(f"Length of feature_sets: {len(feature_sets)}")
    
    for set_name, features in feature_sets.items():
        print(f"Set [{set_name}] includes {len(features)} features.")

if __name__ == "__main__":
    view_features()
