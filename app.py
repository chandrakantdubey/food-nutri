from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import pandas as pd

app = FastAPI()

# Load model
model = torch.jit.load("food_model_ts.pt", map_location=torch.device("cpu"))
model.eval()

# Load class labels
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load nutrition data
df = pd.read_csv("nutrition.csv")  # expects columns: label, weight, calories, protein, carbohydrates, fats, fiber, sugars, sodium

# Create a lookup map for nutrition info by label
nutrition_map = {row['label'].lower(): row.to_dict() for _, row in df.iterrows()}

# Image transform (resize and normalize to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
        predicted_class = classes[predicted.item()]  # e.g., "butter chicken"

    # Nutrition lookup
    nutrition = nutrition_map.get(predicted_class.lower(), {
        "label": predicted_class,
        "message": "Nutrition info not found"
    })

    return JSONResponse(content={
        "class": predicted_class,
        "confidence": round(confidence, 4),
        "nutrition": nutrition
    })
