# In api_server.py
import mindspore as ms
from mindcv.models import create_model 
from mindspore import load_checkpoint, load_param_into_net
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import uvicorn
import time
import datetime
import uuid
import sqlite3
import os
import json
from typing import List, Optional

# --- CORRECTED IMPORTS: Use '.' for relative imports within the 'server' package ---
from .api_utils import preprocess_image, post_process_predictions, CLASSES, NUM_CLASSES 
from .api_models import Classification, ClassificationResult, AnalysisMetadata, AnalysisResult 

# --- Configuration ---
# Assuming you keep the checkpoint path relative to the server folder for simplicity 
MODEL_CKPT_PATH = "server/efficientnet_b1_8class_multilabel_BEST.ckpt" # Or "server/model_data/..." if moved
MODEL_NAME = "efficientnet_b1"
DATABASE_FILE = "traffic_vision_history.db"
UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "exports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)


# --- Database Setup (FIXED: Schema Aligned with Client and Foreign Keys Enabled) ---
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.execute("PRAGMA foreign_keys = ON") # <--- CRITICAL FIX: Enable Foreign Keys
    cursor = conn.cursor()
    
    # 1. analyses table (Main metadata)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            timestamp TEXT,
            image_name TEXT,
            processing_time REAL,
            confidence_threshold REAL,
            batch_id TEXT,             -- Added batch_id to align with client logic/DB
            detections_json TEXT       -- Kept detections_json for legacy/simplicity
        )
    ''')
    
    # 2. classifications table (Required by client's logic/DB structure)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT,
            class_name TEXT,
            confidence REAL,
            FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
            -- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ CRITICAL for Deletion
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Load MindSpore Classification Model (Unchanged) ---
print(f"Loading {MODEL_NAME} model...")
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU") 

api_model = create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
param_dict = load_checkpoint(MODEL_CKPT_PATH)
ms.load_param_into_net(api_model, param_dict)
api_model.set_train(False) 
print(f"Model loaded successfully from {MODEL_CKPT_PATH}")


# --- FastAPI App Definition (Unchanged) ---
app = FastAPI(title="TrafficVision API")

# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
    batch_id: Optional[str] = None # Include batch_id parameter
):
    start_time = time.time()
    image_bytes = await file.read()
    analysis_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or f"upload_{analysis_id}.jpg")
    extension = os.path.splitext(safe_name)[1] or ".jpg"
    stored_filename = f"{analysis_id}{extension}"
    stored_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    with open(stored_path, "wb") as image_file:
        image_file.write(image_bytes)
    image_name = stored_path

    try:
        input_tensor = preprocess_image(image_bytes) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {e}")

    try:
        logits = api_model(input_tensor) 
        results: ClassificationResult = post_process_predictions(logits, confidence_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference or post-processing failed: {e}")

    end_time = time.time()
    processing_time = round(end_time - start_time, 3)
    results.processing_time = processing_time 
    results.confidence_threshold = confidence_threshold

    timestamp = datetime.datetime.now()
    detections_json = results.model_dump_json(exclude={'processing_time', 'confidence_threshold'})

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.execute("PRAGMA foreign_keys = ON") # Ensure foreign keys are ON for consistency
        cursor = conn.cursor()
        
        # 1. Save to analyses table
        cursor.execute(
            "INSERT INTO analyses (analysis_id, timestamp, image_name, processing_time, confidence_threshold, detections_json, batch_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, timestamp.isoformat(), image_name, processing_time, confidence_threshold, detections_json, batch_id)
        )
        
        # 2. Save classifications to separate table (aligning with client logic)
        class_data = [
            (analysis_id, c.class_name, c.confidence)
            for c in results.classifications
        ]
        cursor.executemany(
            "INSERT INTO classifications (analysis_id, class_name, confidence) VALUES (?, ?, ?)",
            class_data,
        )
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"ERROR: Could not save analysis {analysis_id} to database: {e}")

    return AnalysisResult(
        analysisId=analysis_id,
        timestamp=timestamp,
        imageName=image_name,
        batchId=batch_id,
        results=results
    )

@app.get("/history", response_model=List[AnalysisMetadata])
async def get_history():
    """Retrieves basic metadata for all analyses."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row # Use Row factory for named access
        cursor = conn.cursor()
        cursor.execute("SELECT analysis_id, timestamp, image_name, batch_id FROM analyses ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        
        history = [
            AnalysisMetadata(
                analysisId=row['analysis_id'], 
                timestamp=datetime.datetime.fromisoformat(row['timestamp']), 
                imageName=row['image_name'],
                batchId=row['batch_id']
            )
            for row in rows
        ]
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {e}")


@app.get("/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_details(analysis_id: str):
    """Retrieves full analysis details."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. Get main analysis row
        cursor.execute(
            "SELECT analysis_id, timestamp, image_name, processing_time, confidence_threshold, batch_id FROM analyses WHERE analysis_id = ?",
            (analysis_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            conn.close()
            raise HTTPException(status_code=404, detail="Analysis not found")

        # 2. Get classifications from classifications table
        cursor.execute(
            "SELECT class_name, confidence FROM classifications WHERE analysis_id = ?",
            (analysis_id,)
        )
        class_rows = cursor.fetchall()
        
        conn.close()
        
        classifications = [
            Classification(
                className=cr['class_name'],
                confidence=cr['confidence']
            ) for cr in class_rows
        ]
        
        # Reconstruct ClassificationResult
        results = ClassificationResult(
            classifications=classifications,
            processingTime=row['processing_time'],
            confidenceThreshold=row['confidence_threshold'],
        )

        return AnalysisResult(
            analysisId=row['analysis_id'],
            timestamp=datetime.datetime.fromisoformat(row['timestamp']),
            imageName=row['image_name'],
            batchId=row['batch_id'],
            results=results
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis details: {e}")

@app.get("/export/{analysis_id}")
async def export_results(analysis_id: str, format: str = Query("json", enum=["json", "csv", "pdf"])):
    raise HTTPException(status_code=501, detail="Export not implemented yet")

@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Deletes a single analysis from the database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.execute("PRAGMA foreign_keys = ON") # CRITICAL: Ensure cascade works
        cursor = conn.cursor()
        
        # We only need to delete the parent entry. 
        # The ON DELETE CASCADE constraint in the 'classifications' table 
        # definition handles the cleanup of related entries.
        cursor.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Analysis ID not found")
        
        conn.commit()
        conn.close()
        return {"detail": f"Analysis {analysis_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"SERVER ERROR during DELETE: {e}") 
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {e}")

if __name__ == "__main__":
    # Ensure all previous database files are deleted before running this corrected version
    # to avoid schema mismatches if the database file was created with the old schema.
    uvicorn.run(app, host="127.0.0.1", port=8000)