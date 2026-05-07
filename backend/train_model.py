import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

print("=" * 50)
print("🦆 DUCK FARM ML MODEL TRAINING")
print("=" * 50)

# Create sample data if CSV doesn't exist
if not os.path.exists("duck_farm_production_data.csv"):
    print("📝 Creating sample data file...")
    
    sample_data = {
        'Duck_Count': [202, 535, 960, 370, 800, 120, 714, 971, 763, 500],
        'Lay_Eggs': [153, 428, 769, 299, 642, 97, 572, 780, 605, 400],
        'Commercial_Feeds_KG': [14.14, 37.45, 67.2, 25.9, 56, 8.4, 49.98, 67.97, 53.41, 35],
        'Waste_Pellets_KG': [14.14, 37.45, 67.2, 25.9, 56, 8.4, 49.98, 67.97, 53.41, 35],
        'Machine_Input_KG': [15.55, 41.2, 73.92, 28.49, 61.6, 9.24, 54.98, 74.77, 58.75, 38.5],
        'Restaurants_Needed': [5, 11, 20, 8, 16, 3, 15, 20, 16, 10],
        'Egg_Price_PHP': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv("duck_farm_production_data.csv", index=False)
    print("✅ Sample data file created!")

# Load the data
print("\n📊 Loading data...")
df = pd.read_csv("duck_farm_production_data.csv")
print(f"✅ Loaded {len(df)} records")

# Prepare features
X = df[['Duck_Count']]

# Train models
models = {}

print("\n🤖 Training Machine Learning Models...")
print("-" * 30)

# Train Egg Production Model
model_eggs = LinearRegression()
model_eggs.fit(X, df[['Lay_Eggs']])
models['eggs'] = model_eggs
print(f"✅ Egg model trained - R²: {model_eggs.score(X, df[['Lay_Eggs']]):.4f}")

# Train Feed Consumption Model
model_feeds = LinearRegression()
model_feeds.fit(X, df[['Commercial_Feeds_KG']])
models['feeds'] = model_feeds
print(f"✅ Feed model trained - R²: {model_feeds.score(X, df[['Commercial_Feeds_KG']]):.4f}")

# Train Waste Model
model_waste = LinearRegression()
model_waste.fit(X, df[['Waste_Pellets_KG']])
models['waste'] = model_waste
print(f"✅ Waste model trained - R²: {model_waste.score(X, df[['Waste_Pellets_KG']]):.4f}")

# Train Machine Input Model
model_machine = LinearRegression()
model_machine.fit(X, df[['Machine_Input_KG']])
models['machine_input'] = model_machine
print(f"✅ Machine model trained - R²: {model_machine.score(X, df[['Machine_Input_KG']]):.4f}")

# Train Restaurants Model
model_restaurants = LinearRegression()
model_restaurants.fit(X, df[['Restaurants_Needed']])
models['restaurants'] = model_restaurants
print(f"✅ Restaurants model trained - R²: {model_restaurants.score(X, df[['Restaurants_Needed']]):.4f}")

# Save all models
joblib.dump(models, "duck_models.pkl")
print("\n💾 Models saved to: duck_models.pkl")

# Save farm configuration
farm_config = {
    'egg_price': 7.0,
    'feed_cost': 35.0
}
joblib.dump(farm_config, "farm_config.pkl")
print("💾 Config saved to: farm_config.pkl")

print("\n" + "=" * 50)
print("✅ TRAINING COMPLETE!")
print("=" * 50)

# Test prediction
print("\n📈 SAMPLE PREDICTION (500 ducks):")
pred_eggs = int(model_eggs.predict([[500]])[0][0])
pred_feed = round(model_feeds.predict([[500]])[0][0], 2)
print(f"   🥚 Expected eggs: {pred_eggs}")
print(f"   🌾 Feed needed: {pred_feed} KG")
print(f"   💰 Expected income: ₱{pred_eggs * 7}")
print(f"   💸 Feed cost: ₱{pred_feed * 35}")