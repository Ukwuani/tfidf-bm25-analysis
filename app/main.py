from typing import Union

from fastapi import FastAPI, Response, HTTPException, status
from .payloads import Payload
from .main_controller import compute_tfidf_bm25

app = FastAPI()
print("started")
@app.get("/")
def home():
    return "You are Home!"

@app.post("/search")
def read_root(payload: Payload):
    response = compute_tfidf_bm25(payload.query, payload.documents)
    return response


@app.get("/search")
def read_item(query: str):
    response = compute_tfidf_bm25(query)
    return response
