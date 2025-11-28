from transformers import pipeline

# 質問応答モデルをロード
qa = pipeline("question-answering")

# 質問と文脈を与えて回答を取得
result = qa({
    "question": "What is the capital of France?",
    "context": "France is a country in Europe. Its capital is Paris."
})

# 結果を表示
print(result)