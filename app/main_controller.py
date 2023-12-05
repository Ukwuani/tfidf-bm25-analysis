from fastapi import HTTPException
from .tfidf_service import getTfIdfRanking
from .bm25_service import getBM25Ranking
from .payloads import samples

def get_data():
    response = lambda samples: [ ({ 
        "docId": index,
        "document": datum,
        "score": 0
    }) for index, datum in enumerate(samples) ]
    return {
        "tfidf": response(samples),
        "bm25": response(samples),
        }

def compute_tfidf_bm25(query: str = "", document: list[str] = samples):
    try:
        if (not query):
                return get_data()
        tfidf = getTfIdfRanking(document, query)
        bm25 = getBM25Ranking(document, query)
        return {
            "tfidf": tfidf,  
            "bm25": bm25,  
        }
    except:
        # print(Exception)
        return "something went wrong"
    