from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from rest_framework.decorators import api_view
from rest_framework.response import Response

import joblib
import os
from django.conf import settings

# Define paths to the trained model and label encoder
model_path = os.path.join(settings.BASE_DIR, "ml_api", "model.pkl")
encoder_path = os.path.join(settings.BASE_DIR, "ml_api", "label_encoder.pkl")

# Load model and label encoder
model = joblib.load(model_path)

model = joblib.load('ml_api/model.pkl') 
encoder = joblib.load('ml_api/label_encoder.pkl')

label_encoder = joblib.load(encoder_path)

class PredictView(APIView):
    def post(self, request):
        try:
            # Retrieve input values from request data
            math_score = float(request.data.get("math_score"))
            reading_score = float(request.data.get("reading_score"))

            # Make a prediction
            prediction = model.predict([[math_score, reading_score]])
            grade = label_encoder.inverse_transform(prediction)[0]

            return Response({"predicted_grade": grade})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        

        
