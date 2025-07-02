import csv
import json
from sklearn.model_selection import train_test_split

# Load CSV
csv_path = 'src/classification/image_index.csv'
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Separate label 0 and 1
label_0 = [x for x in data if x['label'] == '0']
label_1 = [x for x in data if x['label'] == '1']
other_labels = [x for x in data if x['label'] not in {'0', '1'}]

# Split 80/20 for label 0 and 1
train_0, test_0 = train_test_split(label_0, test_size=0.2, random_state=42)
train_1, test_1 = train_test_split(label_1, test_size=0.2, random_state=42)

# Modify test set: all non-label-1 labels become 0
test_others = []
for x in other_labels:
    test_others.append({'path': x['path'], 'label': 0})

# Combine splits
train_split = train_0 + train_1
test_split = test_0 + test_1 + test_others

# Convert labels to int
for x in train_split + test_split:
    x['label'] = int(x['label'])

# Save JSON
split_json_path = 'src/classification/pepper_vs_all_split.json'
with open(split_json_path, 'w') as f:
    json.dump({'train': train_split, 'test': test_split}, f, indent=2)

print(f"Saved train/test split to {split_json_path}")
print(f"Train size: {len(train_split)}, Test size: {len(test_split)}")
