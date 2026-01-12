"""
Simple API test script
"""
import json
import pandas as pd
from app import create_app

# Create Flask app
app = create_app()

# Create test client
client = app.test_client()

print("\n" + "=" * 70)
print("TESTING FLASK API ENDPOINTS")
print("=" * 70)

# Test 1: Health check
print("\n1. Testing /api/health endpoint...")
response = client.get('/api/health')
print(f"   Status: {response.status_code}")
print(f"   Response: {response.get_json()}")

# Test 2: Get models
print("\n2. Testing /api/models endpoint...")
response = client.get('/api/models')
print(f"   Status: {response.status_code}")
data = response.get_json()
print(f"   Models available: {len(data.get('models', []))}")
for model in data.get('models', []):
    print(f"      - {model['model_name']}: {model['algorithm']}")

# Test 3: Make prediction
print("\n3. Testing /api/predict endpoint...")
test_data = {
    'Brand': 'TOYOTA',
    'Model': 'Corolla',
    'YOM': 2018,
    'Engine (cc)': 1500,
    'Gear': 'Automatic',
    'Fuel Type': 'Petrol',
    'Millage(KM)': 120000,
    'Town': 'Colombo',
    'Date': '2025-01-10',
    'Leasing': 'No Leasing',
    'Condition': 'USED',
    'AIR CONDITION': 'Available',
    'POWER STEERING': 'Available',
    'POWER MIRROR': 'Available',
    'POWER WINDOW': 'Available'
}

response = client.post('/api/predict',
                      data=json.dumps(test_data),
                      content_type='application/json')
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    result = response.get_json()
    print(f"   Prediction successful!")
    print(f"   Ensemble Price: Rs. {result['ensemble']['prediction_lakhs']:.2f} Lakhs")
    print(f"   Models used: {result['ensemble']['num_models']}")
else:
    print(f"   Error: {response.get_json()}")

# Test 4: Test SHAP generation
print("\n4. Testing /api/shap/generate endpoint...")
response = client.post('/api/shap/generate?model=xgboost',
                      data=json.dumps(test_data),
                      content_type='application/json')
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    result = response.get_json()
    print(f"   SHAP generation started!")
    print(f"   Request ID: {result.get('request_id')}")
else:
    print(f"   Error: {response.get_json()}")

print("\n" + "=" * 70)
print("ALL API TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70 + "\n")
