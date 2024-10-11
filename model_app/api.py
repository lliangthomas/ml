from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Optional

import run

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionItem(BaseModel):
    date: str
    actual: Optional[float]
    predicted: Optional[float]

class Response(BaseModel):
    predictions: List[PredictionItem]

@app.get("/prediction", response_model=Response)
async def get_predictions():
    try:
        predictions = run.inference(name="hybrid_lr_0.001_window_120_epoch_1000")
        formatted_predictions = [
            PredictionItem(
                date=value["date"],
                actual=value.get("actual"), # none if not there
                predicted=value.get("prediction") # none if not there
            )
            for value in predictions
        ]
        return Response(predictions=formatted_predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)