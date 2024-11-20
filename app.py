import pandas as pd
from Preprocessing_EN import Preprocessing_EN
import pyterrier as pt
if not pt.started():
    pt.init()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 



lawyers = pd.read_csv(r"Data\lawyers.csv").rename(columns={
    "id":"lawyer_id"
}).drop(columns={"email","email_verified_at","password","remember_token","phone","affiliation_date","created_at","updated_at"})
# lawyers.head()

lawyers['years_of_experience'] = lawyers['years_of_experience'].apply(lambda x: f"{x}year")
# lawyers.head()

rates = pd.read_csv(r"Data\rates.csv").drop(columns={"created_at","updated_at","id"})
# rates.head()

rates['rating'] = rates['rating'].apply(lambda x :f"{x}star")
# rates.head()


lawyers_with_rates = lawyers.merge(rates,on="lawyer_id")

lawyers_with_rates["Text"] = lawyers_with_rates[['name','address','union_branch','years_of_experience','rating','review']].astype(str).agg(" ".join,axis=1)

lawyers_with_rates['processed_text'] = lawyers_with_rates['Text'].apply(Preprocessing_EN.process)

documents = lawyers_with_rates[['lawyer_id','processed_text']].copy()
documents['processed_text'] = documents['processed_text'].apply(lambda x:" ".join(x))
documents.rename(columns={
    "lawyer_id":"docno",
    "processed_text":"text"
},inplace=True)

index_path = r'C:\Users\Mohammad Kher\Desktop\Projects\IR\lawyer_index'
indexer = pt.DFIndexer(index_path,overwrite=True)

index_ref = indexer.index(documents['text'],documents['docno'].astype(str))

print("Index created with reference:", index_ref)


# bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
# query = "experienced lawyer"
# results = bm25.search(query)
# print(results[['docno', 'score']])



def Top_Recommendations(user_input,lawyer_data):
    all_descriptions = [user_input] + lawyer_data["text"].tolist()
    # all_descriptions

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    user_vector = tfidf_matrix[0]  
    lawyer_vectors = tfidf_matrix[1:]
    
    similarities = cosine_similarity(user_vector, lawyer_vectors).flatten()

    ranked_lawyers = np.argsort(similarities)[::-1]

    final_result = {}

    for idx in ranked_lawyers:
        lawyer_id = lawyer_data.iloc[idx]["docno"]
        description = lawyer_data.iloc[idx]["text"]
        similarity = similarities[idx]

        final_result[lawyer_id] = [f"{similarity:.2f}",description]
        
        # final_result.append(f"Lawyer ID: {lawyer_id}, Similarity: {similarity:.2f}, Description: {description}")
    return final_result

user_input = "experienced lawyer"
print(Top_Recommendations(user_input,documents))