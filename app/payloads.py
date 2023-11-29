from pydantic import BaseModel

samples = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
    "all foods are nice"
]

class Payload(BaseModel):
    query: str
    documents: list[str] = samples