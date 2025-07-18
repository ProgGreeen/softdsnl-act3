import joblib

# Load the trained model and label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Sample input: [math_score, reading_score]
sample = [[86, 88]]  # You can modify this with other score pairs

# Make prediction
prediction = model.predict(sample)
predicted_grade = label_encoder.inverse_transform(prediction)[0]

print(f"📘 Predicted Grade for Math={sample[0][0]} and Reading={sample[0][1]}: {predicted_grade}")
