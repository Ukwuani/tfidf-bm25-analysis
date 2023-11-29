from fastapi import HTTPException
from .tfidf_service import getTfIdfRanking
from .bm25_service import getBM25Ranking
from .payloads import samples

def compute_tfidf_bm25(query: str, document: list[str] = samples):
    try:
        if (not query):
                raise HTTPException(status_code=401, detail="Provide a text as query")
        tfidf = getTfIdfRanking(document, query)
        bm25 = getBM25Ranking(document, query)
        return {
                "tfidf": tfidf,  
                "bm25": bm25,  
        }
    except:
        # print(Exception)
        return "something went wrong"