#!/bin/bash
echo "🚀 Setting up Smart Data Agent..."

# Create structure
mkdir -p modules
touch modules/__init__.py

# Install requirements
pip install -r requirements.txt

# Create sample data
python -c "
import pandas as pd
import numpy as np
np.random.seed(42)
df = pd.DataFrame({
    'id': range(1000),
    'price': np.random.uniform(10,500,1000),
    'rating': np.random.uniform(1,5,1000)
})
df.to_csv('sample_data.csv', index=False)
print('✅ Sample data created!')
"

echo "✅ Setup complete! Run: streamlit run app.py"
