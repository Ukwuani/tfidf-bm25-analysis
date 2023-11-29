from rank_bm25 import BM25Okapi

def getBM25Ranking(documents, query):
    # Tokenize the query and documents 
    tokenizedQuery = query.split()
    tokenizedDocuments = [doc.split() for doc in documents]

    # Create a BM25 model
    bm25 = BM25Okapi(tokenizedDocuments)

    # Get the BM25 scores for each document
    bm25Scores = bm25.get_scores(tokenizedQuery)

    indexes = bm25Scores.argsort()[::-1]
    
    retrievedDocs = []
    for i in indexes:
        retrievedDocs.append({
            "docId": int(i+1),
            "document": documents[i],
            "score": bm25Scores[i],
            })
    return retrievedDocs

