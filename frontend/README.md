# Car Price Prediction Frontend

React + TypeScript + Vite frontend for car price prediction with SHAP explanations.

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Backend server running on http://localhost:5000

## Installation

```bash
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The application will be available at: http://localhost:5173

## Build

Create a production build:

```bash
npm run build
```

## Preview Production Build

Preview the production build:

```bash
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── PredictionForm/       # Input form component
│   ├── PredictionResults/    # Results display component
│   ├── ShapExplanation/      # SHAP visualizations
│   └── common/               # Reusable components
├── services/
│   ├── api.ts                # Axios configuration
│   └── predictionService.ts  # API calls
├── types/
│   └── index.ts              # TypeScript types
├── utils/
│   ├── constants.ts          # Constants and options
│   └── validation.ts         # Validation logic
├── App.tsx                   # Root component
├── App.css                   # Global styles
└── main.tsx                  # Entry point
```

## Features

- Interactive car price prediction form
- Real-time validation
- Ensemble predictions from multiple ML models
- SHAP explanations with visualizations
- Responsive design
- Error handling and loading states

## API Integration

The frontend connects to the Flask backend via proxy configuration in `vite.config.ts`:

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:5000',
    changeOrigin: true,
  }
}
```

## Technologies

- React 19
- TypeScript
- Vite
- Axios
- CSS Modules
