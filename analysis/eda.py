import json
import os

def view_features():
    features_path = 'data/features.json'
    
    if not os.path.exists(features_path):
        pass

    with open(features_path, 'r') as f:
        data = json.load(f)

    feature_sets = data.get('feature_sets', {})
    print(f"Length of feature_sets: {len(feature_sets)}")
    
    for set_name, features in feature_sets.items():
        print(f"Set [{set_name}] includes {len(features)} features.")

if __name__ == "__main__":
    view_features()