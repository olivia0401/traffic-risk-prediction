"""Clean Jupyter Notebook by removing error cells and warnings"""
import json

# Read the notebook
with open('/home/olivia/traffic-risk-prediction/notebooks/traffic_risk_eda.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell IDs to remove (cells with errors)
cells_to_remove = [
    '1a5d8b0e-d464-484e-a871-50d7359be6e2',  # NameError cell
    'b31dfaff-9865-448b-a41d-a1af763b092b'   # Empty cell at end
]

# Filter out error cells
nb['cells'] = [cell for cell in nb['cells'] if cell.get('id') not in cells_to_remove]

# Clean DtypeWarning outputs from cell 59e4710d-872b-4b8f-846b-d02db1f4162e
for cell in nb['cells']:
    if cell.get('id') == '59e4710d-872b-4b8f-846b-d02db1f4162e':
        # Remove stderr outputs with DtypeWarning
        if 'outputs' in cell:
            cell['outputs'] = [
                output for output in cell['outputs']
                if output.get('name') != 'stderr'
            ]

# Save cleaned notebook
with open('/home/olivia/traffic-risk-prediction/notebooks/traffic_risk_eda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook cleaned successfully")
print(f"✓ Removed {len(cells_to_remove)} error cells")
print(f"✓ Total cells remaining: {len(nb['cells'])}")
