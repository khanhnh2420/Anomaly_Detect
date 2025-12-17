from fastapi import FastAPI, UploadFile
import tempfile
import os
import pandas as pd
from src.pipeline import run_pipeline

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile):
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        # Read the uploaded file content
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # The run_pipeline function should be updated to return only 2 values (scores, preds)
        # Assuming it returns final_score and y_pred
        scores, preds = run_pipeline(tmp_path)
    finally:
        # Ensure the temporary file is deleted after processing
        os.remove(tmp_path)

    return {
        "total_records": len(preds),
        "anomalies": int(preds.sum())
    }
