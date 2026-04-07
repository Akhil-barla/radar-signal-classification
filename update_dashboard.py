import json
import re

with open('outputs/radar_results.json', 'r') as f:
    data = json.load(f)

for r in data['results']:
    if r['model'] == 'FNN': r['color'] = '#FF6B6B'
    elif r['model'] == 'CNN': r['color'] = '#FFD93D'
    elif r['model'] == 'LSTM': r['color'] = '#00FFD1'

results_str = json.dumps(data['results'], indent=2)

histories_str = "{\n"
for model, h in data['histories'].items():
    histories_str += f"  {model}: {{ loss: {h['loss']}, val_accuracy: {h['val_accuracy']} }},\n"
histories_str += "}"

cm_str = json.dumps(data['confusion_matrices'], indent=2)

with open('radar_dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace RESULTS block - simpler regex to match array with trailing commas
html = re.sub(r'const RESULTS = \[.*?\];', f'const RESULTS = {results_str};', html, flags=re.DOTALL)
# Replace HISTORIES block
html = re.sub(r'const HISTORIES = \{.*?\};', f'const HISTORIES = {histories_str};', html, flags=re.DOTALL)
# Replace CM_DATA block
html = re.sub(r'const CM_DATA = \{.*?\};', f'const CM_DATA = {cm_str};', html, flags=re.DOTALL)

with open('radar_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard updated successfully.")
