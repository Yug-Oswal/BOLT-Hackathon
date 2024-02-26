from .db import get_db
from sqlite3 import Error
from fastapi import HTTPException
from typing_extensions import Tuple

def compute_emotion_and_save(llm_response: str, body_text: str):
    with get_db() as conn:
        emotion = determine_emotion(body_text)
        try:
            conn.execute("INSERT INTO MyTable (text, emotion1, emotion2, emotion3, response) VALUES (?, ?, ?, ?, ?)", (body_text, emotion[0], emotion[1], emotion[2], llm_response))
            conn.commit()
        except Error as e:
            print(e)
            raise HTTPException(status_code=500, detail="Database error")        

def determine_emotion(text: str) -> Tuple[str, str, str]:
    from transformers import pipeline
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    outputs = classifier(text)
    return outputs[0][0]['label'], outputs[0][1]['label'], outputs[0][2]['label']  # type: ignore