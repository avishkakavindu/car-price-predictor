# Car Price Prediction Web Application

A full-stack machine learning web application for predicting car prices in Sri Lanka using ensemble models with SHAP explanations.

## Team Members

| Name | Student ID |
|------|------------|
| Perera A.A.R.D. | 258812E |
| Perera W.A.N.M. | 258815P |
| Dambawinna W.R.P.W.M.A.K.B. | 258765K |

## Features

- **Multiple ML Models**: XGBoost, LightGBM, and AdaBoost with extensible architecture
- **Ensemble Predictions**: Combines predictions from all models with confidence metrics
- **SHAP Explanations**:
  - Feature importance bar charts
  - Waterfall plots for individual predictions
  - Detailed interpretation guides
- **Modern UI**: React + TypeScript frontend with responsive design
- **Professional Architecture**: Factory/Strategy patterns for scalability
- **Real-time Validation**: Client and server-side input validation
- **Async Processing**: Non-blocking SHAP generation

## Tech Stack

### Backend
- Python 3.8+
- Flask (REST API)
- XGBoost, LightGBM, AdaBoost (ML models)
- SHAP (Model explanations)
- Pandas, NumPy, scikit-learn
- Matplotlib (Visualizations)

### Frontend
- React 19
- TypeScript
- Vite (Build tool)
- Axios (HTTP client)
- CSS Modules

## Project Structure

```
.
├── backend/
│   ├── app.py                      # Flask application entry point
│   ├── config.py                   # Configuration
│   ├── requirements.txt            # Python dependencies
│   ├── api/                        # REST API endpoints
│   ├── models/                     # ML model classes (Factory pattern)
│   ├── services/                   # Business logic & SHAP service
│   ├── utils/                      # Utilities & error handlers
│   └── data/
│       ├── models/                 # Trained model files (.pkl)
│       └── datasets/               # Training data
│
├── frontend/
│   ├── src/
│   │   ├── components/             # React components
│   │   ├── services/               # API integration
│   │   ├── types/                  # TypeScript types
│   │   ├── utils/                  # Utilities & validation
│   │   ├── App.tsx                 # Root component
│   │   └── main.tsx                # Entry point
│   ├── package.json
│   └── vite.config.ts
│
└── PROJECT_PLAN.md                 # Detailed implementation plan
```

## Prerequisites

### Backend
- Python 3.8 or higher
- pip (Python package manager)

### Frontend
- Node.js 16 or higher
- npm or yarn

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "1. Assignment"
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Important**: The backend requires the trained model files to be placed in `backend/data/models/`:
- `car_price_predictor_xgbr.pkl` (XGBoost)
- `car_price_predictor_lgbm.pkl` (LightGBM)
- `car_price_predictor_adaboost.pkl` (AdaBoost)

### 3. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install
```

## Running the Application

### Start Backend Server

```bash
# From backend directory (with virtual environment activated)
cd backend
python app.py
```

The backend will start on: **http://localhost:5000**

You should see:
```
[OK] Model loaded: XGBoost
[OK] Model loaded: LightGBM
[OK] Model loaded: AdaBoost
Flask app running on http://localhost:5000
Models loaded: 3
Available models: ['xgboost', 'lightgbm', 'adaboost']
```

### Start Frontend Development Server

Open a **new terminal** and run:

```bash
# From frontend directory
cd frontend
npm run dev
```

The frontend will start on: **http://localhost:5173**

### Access the Application

Open your browser and navigate to: **http://localhost:5173**

## Using the Application

### 1. Fill in Car Details

Enter the following information:
- **Brand**: Select from dropdown (Toyota, Honda, Nissan, etc.)
- **Model**: Enter car model name (e.g., Corolla, Civic)
- **Year of Manufacture**: Year between 1990 and current year
- **Engine (cc)**: Engine capacity (500-5000)
- **Gear**: Automatic or Manual
- **Fuel Type**: Petrol, Diesel, Hybrid, or Electric
- **Mileage (KM)**: Total mileage (0-500,000)
- **Town**: Select city in Sri Lanka
- **Condition**: NEW or USED
- **Leasing**: Leasing status
- **Features**: Air Condition, Power Steering, Power Mirror, Power Window

### 2. Get Prediction

Click **"Get Price Prediction"** button

### 3. View Results

The application displays:
- **Ensemble Prediction** (highlighted): Average price with confidence range
- **Individual Model Predictions**: Predictions from each ML model
- **Price Range Visualization**: Min/max range with ensemble marker
- **Input Summary**: Your submitted car details

### 4. SHAP Explanations (Appears after 5-10 seconds)

- **Feature Importance Chart**: Top features affecting prices globally
- **Top 10 Features Table**: Ranked by importance with visual bars
- **Waterfall Plot**: Step-by-step price calculation for your specific car
- **Interpretation Guide**: How to read the visualizations

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check and model status |
| `/api/models` | GET | Get available models info |
| `/api/predict` | POST | Get predictions from all models |
| `/api/shap/generate` | POST | Start SHAP generation (async) |
| `/api/shap/status/:id` | GET | Poll SHAP generation status |

## Testing

### Backend Testing

```bash
# From backend directory
cd backend
python test_api.py
```

This tests all API endpoints with sample data.

### Frontend Build Test

```bash
# From frontend directory
cd frontend
npm run build
```

### Production Build

```bash
# Backend: Run normally with python app.py

# Frontend: Build and preview
cd frontend
npm run build
npm run preview
```

## Adding New Models

The architecture supports easy addition of new ML models:

1. **Train and save the model**:
```python
model_pipeline.fit(X_train, y_train)
joblib.dump(model_pipeline, 'backend/data/models/new_model.pkl')
```

2. **Create model class** (`backend/models/new_model.py`):
```python
from .base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, model_path: str):
        super().__init__(model_name="New Model", model_path=model_path)
```

3. **Register in factory** (`backend/models/model_factory.py`):
```python
MODEL_REGISTRY = {
    'xgboost': {...},
    'new_model': {
        'class': NewModel,
        'path': 'data/models/new_model.pkl'
    }
}
```

The system automatically includes new models in ensemble predictions and UI!

## Troubleshooting

### Backend Issues

**Issue**: `ModuleNotFoundError` when running backend
- **Solution**: Ensure virtual environment is activated and dependencies installed

**Issue**: Model loading error
- **Solution**: Check that `car_price_predictor_xgbr.pkl` exists in `backend/data/models/`

**Issue**: scikit-learn version warning
- **Solution**: The app uses scikit-learn 1.5.2 for compatibility. This is intentional.

**Issue**: Port 5000 already in use
- **Solution**: Change port in `backend/app.py` and `frontend/vite.config.ts` proxy

### Frontend Issues

**Issue**: Port 5173 already in use
- **Solution**: Vite will automatically suggest an alternative port

**Issue**: API connection error
- **Solution**: Ensure backend is running on http://localhost:5000

**Issue**: SHAP not appearing
- **Solution**: Wait 5-10 seconds for async generation. Check backend console for errors.

### Common Issues

**Issue**: CORS errors
- **Solution**: Backend has CORS enabled. Ensure backend is running and accessible.

**Issue**: Build fails with TypeScript errors
- **Solution**: Run `npm install` again and ensure all dependencies are installed

## Development

### Backend Development

```bash
# Enable debug mode in backend/app.py
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Frontend Development

Hot module replacement (HMR) is enabled by default with Vite.

## Environment Variables (Optional)

Create `.env` files for environment-specific configuration:

### Backend `.env`
```
FLASK_ENV=development
FLASK_DEBUG=True
MODEL_PATH=data/models/
```

### Frontend `.env`
```
VITE_API_BASE_URL=http://localhost:5000/api
```

## Performance Notes

- First SHAP generation takes ~5-10 seconds
- Predictions are fast (<1 second)
- Frontend build creates optimized production bundle
- Backend uses ThreadPoolExecutor for async SHAP processing

## Architecture Highlights

- **Strategy Pattern**: BaseModel interface for all ML models
- **Factory Pattern**: Centralized model loading and management
- **Service Layer**: Separation of business logic from API layer
- **Async Processing**: Non-blocking SHAP generation
- **Type Safety**: Full TypeScript coverage in frontend
- **Component Architecture**: Reusable React components

## Assignment Compliance

This project fulfills the MSc AI Machine Learning Assignment requirements:

✓ Trained ML models integrated (XGBoost, LightGBM, AdaBoost)
✓ User input functionality (14 features)
✓ Ensemble prediction display with confidence metrics
✓ SHAP explanations (bar chart + waterfall plot)
✓ Professional React + Flask implementation
✓ Bonus features: Multiple models, async SHAP, price range visualization, caching

## Credits

- **ML Models**: XGBoost, LightGBM, AdaBoost with preprocessing pipelines
- **SHAP**: Model interpretability
- **Dataset**: Car price dataset from Sri Lanka

## License

This is an academic project for MSc in AI coursework.

## Contact

For issues or questions, please refer to the PROJECT_PLAN.md for detailed implementation documentation.
