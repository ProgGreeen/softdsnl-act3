import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the dataset
df = pd.read_csv("my_dataset.csv")

# 2. Visualize the data
sns.scatterplot(data=df, x="math_score", y="reading_score", hue="grade", palette="Set2")
plt.title("Student Scores and Grade Distribution")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.grid(True)
plt.show()

# 3. Prepare features and labels
X = df[["math_score", "reading_score"]]
le = LabelEncoder()
y = le.fit_transform(df["grade"])

# 4. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 5. Save model and label encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model trained and saved using my_dataset.csv.")
