# In api_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

# --- NEW: A single classification result ---
class Classification(BaseModel):
    class_name: str = Field(..., alias="className")
    confidence: float

# --- MODIFIED: This replaces DetectionResults ---
class ClassificationResult(BaseModel):
    # This is now a list of classifications, NOT detections
    classifications: List[Classification]
    processing_time: float = Field(..., alias="processingTime")
    confidence_threshold: Optional[float] = Field(None, alias="confidenceThreshold")

# --- MODIFIED: This stays mostly the same but uses the new result type ---
class AnalysisMetadata(BaseModel):
    analysis_id: str = Field(..., alias="analysisId")
    timestamp: datetime
    image_name: str = Field(..., alias="imageName")
    batch_id: Optional[str] = Field(None, alias="batchId")

class AnalysisResult(AnalysisMetadata):
    # This now holds a ClassificationResult
    results: ClassificationResult

# --- You can DELETE BoundingBox and Detection, or keep them for later ---