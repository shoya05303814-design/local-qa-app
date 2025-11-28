import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from transformers import pipeline
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import SessionLocal, Base, engine, QAHistory
import logging

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================
#  .env 読み込み
# ==============================
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") # 英語モデル
JA_MODEL = os.getenv("JA_MODEL")           # 日本語モデル

# DB初期化
Base.metadata.create_all(bind=engine)

# ==============================================================================
# FastAPI初期化
# ==============================================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# モデル読み込み（自動読み込み）
# ==============================
print("Loading English model:", DEFAULT_MODEL)
qa_en = pipeline("question-answering", model=DEFAULT_MODEL)

print("Loading Japanese model:", JA_MODEL)
qa_ja = pipeline("question-answering", model=JA_MODEL)

# ==============================
# API 入出力モデル
# ==============================
class AskRequest(BaseModel):
    question: str
    context: str = ""
    model: str = "" # フロントから選択されたモデル名を受け取る

# ---------------------
# /ask 質問処理
# ---------------------
@app.post("/ask")
def ask_question(req: AskRequest):
    logger.info(f"Received question: {req.question}")

    # モデル選択
    try:
        if req.model:
            selected_model = req.model
        elif any(c in req.question for c in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよわをん"):
            selected_model = JA_MODEL
        else:
            selected_model = DEFAULT_MODEL

        qa_pipeline = qa_ja if selected_model == JA_MODEL else qa_en

    except Exception as e:
        logger.error(f"Model selection error: {str(e)}")
        return {"error": "Model selection failed"}

    # 推論処理
    try:
        result = qa_pipeline(
            question=req.question,
            context=req.context or " "
        )
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return {"error": "Model failed to answer"}

    # DB 保存
    try:
        db: Session = SessionLocal()
        history = QAHistory(
            question=req.question,
            answer=result.get("answer"),
            confidence=f"score={result.get('score'):.3f}",
        )
        db.add(history)
        db.commit()
    except Exception as e:
        logger.error(f"DB insert error: {str(e)}")

    logger.info("Answer generated successfully.")

    return {
        "answer": result.get("answer"),
        "confidence": f"score={result.get('score'):.3f}",
        "model_used": selected_model,
    }

# ---------------------
# /history 履歴取得
# ---------------------
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

# --- index.html を返す設定 ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")