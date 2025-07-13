from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import io
from llm_evaluator import LLMEvaluator

app = FastAPI()

_evaluator = None

@app.post('/initialize')
def initialize(api_key: str = Form(...), model: str = Form('gpt-4o-mini')):
    global _evaluator
    _evaluator = LLMEvaluator(api_key=api_key, model=model)
    return {"status": "initialized"}

@app.post('/predict')
async def predict(file: UploadFile = File(...), prompt: str = Form(...), target_column: str = Form(...)):
    if _evaluator is None:
        return {"error": "not initialized"}

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    classes = _evaluator.extract_classes_from_data(df, target_column)
    predictions = _evaluator.evaluate_batch_parallel(
        df,
        prompt,
        classes,
        [col for col in df.columns if col != target_column]
    )
    return {"predictions": predictions}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
