import io
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.linear_model import LinearRegression
import os

# Create FastAPI app
app = FastAPI(title="Duck Farm ML API")

# Enable CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models = {}
farm_config = {"egg_price": 7.0, "feed_cost": 35.0}

# Load existing models if they exist
if os.path.exists("duck_models.pkl"):
    try:
        models = joblib.load("duck_models.pkl")
        farm_config = joblib.load("farm_config.pkl")
        print("✅ Models loaded successfully")
        print(f"   Available models: {list(models.keys())}")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
else:
    print("⚠️ No models found. Please upload CSV file first.")

@app.get("/")
def home():
    """Check if API is running"""
    return {
        "message": "🦆 Duck Farm Production API is online",
        "status": "running",
        "models_ready": len(models) > 0
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV and train ML models"""
    global models, farm_config
    
    try:
        # Read the CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        print(f"📊 Received {len(df)} records for training")
        
        # Update egg price if available
        if 'Egg_Price_PHP' in df.columns:
            farm_config["egg_price"] = float(df['Egg_Price_PHP'].iloc[-1])
        
        # Prepare features
        X = df[['Duck_Count']]
        
        # Train models for each target
        targets = {
            'eggs': 'Lay_Eggs',
            'feeds': 'Commercial_Feeds_KG',
            'waste': 'Waste_Pellets_KG',
            'machine_input': 'Machine_Input_KG',
            'restaurants': 'Restaurants_Needed'
        }
        
        trained_count = 0
        for model_name, column_name in targets.items():
            if column_name in df.columns:
                model = LinearRegression()
                model.fit(X, df[[column_name]])
                models[model_name] = model
                trained_count += 1
                print(f"   ✅ Trained {model_name} model")
        
        # Save models
        joblib.dump(models, "duck_models.pkl")
        joblib.dump(farm_config, "farm_config.pkl")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Trained {trained_count} models successfully",
            "models": list(models.keys()),
            "records": len(df)
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def predict(duck_count: int, money_on_hand: float = 0):
    """Get predictions for a specific duck count"""
    
    # Check if models are trained
    if not models:
        raise HTTPException(
            status_code=400, 
            detail="Models not trained. Please upload CSV file first."
        )
    
    try:
        # Ensure duck count is at least 1
        duck_count = max(1, duck_count)
        
        # Prepare input
        X_input = np.array([[duck_count]])
        
        # Make predictions
        pred_eggs = int(max(0, models['eggs'].predict(X_input)[0][0]))
        pred_feeds = round(float(models['feeds'].predict(X_input)[0][0]), 2)
        pred_waste = round(float(models['waste'].predict(X_input)[0][0]), 2)
        pred_machine = round(float(models['machine_input'].predict(X_input)[0][0]), 2)
        pred_restaurants = int(np.ceil(models['restaurants'].predict(X_input)[0][0]))
        
        # Calculate finances
        egg_price = farm_config.get("egg_price", 7.0)
        feed_cost = farm_config.get("feed_cost", 35.0)
        
        income = pred_eggs * egg_price
        expenses = pred_feeds * feed_cost
        net_income = income - expenses
        
        # Calculate profit margin
        profit_margin = (net_income / income * 100) if income > 0 else 0
        
        # Return results
        return {
            "duck_count": duck_count,
            "operational_forecast": {
                "predicted_eggs": pred_eggs,
                "commercial_feeds_kg": pred_feeds,
                "waste_pellets_kg": pred_waste,
                "machine_input_kg": pred_machine,
                "restaurants_needed": pred_restaurants,
                "eggs_per_duck": round(pred_eggs / duck_count, 2)
            },
            "financial_forecast": {
                "income": round(income, 2),
                "expenses": round(expenses, 2),
                "net_income": round(net_income, 2),
                "profit_margin": round(profit_margin, 1),
                "starting_balance": round(money_on_hand, 2),
                "ending_balance": round(money_on_hand + net_income, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if API is healthy"""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "model_count": len(models)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("🚀 STARTING FASTAPI SERVER")
    print("=" * 50)
    print("Server will run at: http://0.0.0.0:8000")
    print("Press Ctrl+C to stop")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)