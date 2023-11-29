from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def getTfIdfRanking(documents, query):
    vectorizer = TfidfVectorizer()
    
    # Step 2: Fit and transform the documents
    tfidfMatrix = vectorizer.fit_transform(documents)
    
    # Step 3: Transform the query using the same vectorizer
    queryTfIdf = vectorizer.transform([query])

    # Step 4: Calculate cosine similarity between the query and documents
    cosineSimilarities = cosine_similarity(queryTfIdf, tfidfMatrix)
    
    indexes = cosineSimilarities[0].argsort()[::-1]
    cosineSimilarities = cosineSimilarities[0].tolist()
    retrievedDocs = []
    for i in indexes:
        retrievedDocs.append({
            "docId": int(i+1),
            "document": documents[i],
            "score": cosineSimilarities[i],
            })
    return retrievedDocs
