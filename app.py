from fastapi import FastAPI, UploadFile
import pandas as pd
from src.pipeline import run_pipeline

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile):
    df = pd.read_csv(file.file)
    df.to_csv("temp.csv", index=False)

    scores, preds = run_pipeline("temp.csv")

    return {
        "total_records": len(preds),
        "anomalies": int(preds.sum())
    }
