
import pickle
from pathlib import Path

from model import normalize_record

# Load the trained model
model_path = Path(__file__).with_name("pipeline.bin")

with model_path.open("rb") as f:
    pipeline = pickle.load(f)

# Your lead data
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

# Apply the same preprocessing as your FastAPI endpoint
record = normalize_record(lead)

# Calculate conversion probability
probabilities = pipeline.predict_proba([record])[0]
class_index = list(pipeline.classes_).index(1)
probability = float(probabilities[class_index])

print(f"Conversion probability: {probability:.3f}")
