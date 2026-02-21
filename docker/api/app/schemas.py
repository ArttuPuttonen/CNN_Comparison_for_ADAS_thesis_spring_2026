from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    model: str = Field(..., description="Model name")
    class_id: str = Field(..., description="Predicted class index as string")
    confidence: str = Field(..., description="Top class confidence score as string")
    inference_time_ms: str = Field(..., description="Inference latency in milliseconds as string")


class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Original uploaded filename")
    predictions: list[PredictionItem] = Field(..., description="Predictions for all configured models")
