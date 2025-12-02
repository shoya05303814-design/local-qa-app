import os
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import SessionLocal, Base, engine, QAHistory
import logging
from transformers import AutoTokenizer  # tokenizer は軽いので OK

# -----------------------------------
# Logging
# -----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------
# Load Env
# -----------------------------------
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")  # 英語モデル（例：distilbert-base-cased-distilled-squad）
JA_MODEL = os.getenv("JA_MODEL")            # 日本語モデル（例：cl-tohoku/bert-base-japanese-whole-word-masking）

# -----------------------------------
# DB init
# -----------------------------------
Base.metadata.create_all(bind=engine)

# -----------------------------------
# FastAPI init
# -----------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Helper：ONNX モデルをロードして QA を実行するクラス
# -----------------------------------
class ONNXPipeline:
    def __init__(self, model_name):
        # ONNX モデルファイルパス（例：models/distilbert-base.onnx）
        model_path = f"models/{model_name}.onnx"

        # tokenizer は transformers で問題なし（軽量）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ONNX Runtime セッション
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def __call__(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="np")

        # モデル実行
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )

        start_logits, end_logits = outputs

        # 予測スコアから answer を抽出
        start = int(np.argmax(start_logits))
        end = int(np.argmax(end_logits)) + 1

        tokens = inputs["input_ids"][0][start:end]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)

        score = float(
            np.max(start_logits) + np.max(end_logits)
        )  # 簡易スコア

        return {"answer": answer, "score": score}

# -----------------------------------
# Load ONNX models
# -----------------------------------
logger.info("Loading English ONNX model...")
qa_en = ONNXPipeline(DEFAULT_MODEL)

logger.info("Loading Japanese ONNX model...")
qa_ja = ONNXPipeline(JA_MODEL)

# -----------------------------------
# Request schema
# -----------------------------------
class AskRequest(BaseModel):
    question: str
    context: str = ""
    model: str = ""


# -----------------------------------
# /ask
# -----------------------------------
@app.post("/ask")
def ask_question(req: AskRequest):

    # モデル選択
    if req.model:
        selected_model = req.model
    elif any(c in req.question for c in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよわをん"):
        selected_model = JA_MODEL
    else:
        selected_model = DEFAULT_MODEL

    qa_pipeline = qa_ja if selected_model == JA_MODEL else qa_en

    # 推論
    result = qa_pipeline(req.question, req.context or " ")

    # DB保存
    try:
        db: Session = SessionLocal()
        history = QAHistory(
            question=req.question,
            answer=result["answer"],
            confidence=f"score={result['score']:.3f}",
        )
        db.add(history)
        db.commit()
    except Exception as e:
        logger.error(f"DB insert failed: {e}")

    return {
        "answer": result["answer"],
        "confidence": f"score={result['score']:.3f}",
        "model_used": selected_model,
    }


# -----------------------------------
# /history
# -----------------------------------
@app.get("/history")
def get_history():
    db: Session = SessionLocal()
    items = db.query(QAHistory).order_by(QAHistory.id.desc()).all()

    return [
        {
            "id": item.id,
            "question": item.question,
            "answer": item.answer,
            "created_at": item.created_at,
        }
        for item in items
    ]


# -----------------------------------
# Static files + index
# -----------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")