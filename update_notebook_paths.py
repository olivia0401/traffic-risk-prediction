"""Update Notebook to use 2025 data files"""
import json

# Read the notebook
with open('/home/olivia/traffic-risk-prediction/notebooks/traffic_risk_eda.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the data loading cell (cell id: 59e4710d-872b-4b8f-846b-d02db1f4162e)
for cell in nb['cells']:
    if cell.get('id') == '59e4710d-872b-4b8f-846b-d02db1f4162e':
        # Update source code to use 2025 data
        cell['source'] = [
            "#Load CSV - 2025 UK DfT Data\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "from IPython.display import display\n",
            "\n",
            "collision = pd.read_csv(\"../data/collision_2025.csv\", low_memory=False)\n",
            "casualty = pd.read_csv(\"../data/casualty_2025.csv\", low_memory=False)\n",
            "vehicle = pd.read_csv(\"../data/vehicle_2025.csv\", low_memory=False)"
        ]
        # Clear outputs to remove DtypeWarning
        cell['outputs'] = []
        cell['execution_count'] = None
        print(f"✓ Updated cell {cell['id']}")

# Update title cell to mention 2025
for cell in nb['cells']:
    if cell.get('id') == 'ca92a23a-a43d-4e28-8241-e4264a44f17a':
        cell['source'][0] = '# **Predictive Traffic Risk Modelling for Autonomous Driving (2025 Data)**\n'
        print(f"✓ Updated title cell")

# Save updated notebook
with open('/home/olivia/traffic-risk-prediction/notebooks/traffic_risk_eda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✓ Notebook updated successfully for 2025 data")
print("✓ Data paths now point to: ../data/collision_2025.csv, etc.")
print("✓ Added low_memory=False to suppress warnings")
