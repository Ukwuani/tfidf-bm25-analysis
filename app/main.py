from typing import Union

from fastapi import FastAPI, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from .payloads import Payload
from .main_controller import compute_tfidf_bm25, get_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("started")

@app.get("/")
def home():
    response = get_data()
    return response

@app.post("/search")
def read_root(payload: Payload):
    response = compute_tfidf_bm25(payload.query, payload.documents)
    return response


@app.get("/search")
def read_item(query: str = ""):
    response = compute_tfidf_bm25(query)
    return response
