import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Step 1: Load and Prepare Data
def load_and_prepare_data(file_path, num_docs=100):
    df = pd.read_csv(file_path)
    df['combined_text'] = df['title'] + " " + df['description'] + " " + df['requirements']
    return df['combined_text'].head(num_docs).tolist()

# Step 2: Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()
    tokens = text.split()  # Tokenize
    stopwords = set(open("stopwords.txt").read().split())  
    tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords
    return " ".join(tokens)

def preprocess_collection(collection):
    return [preprocess_text(doc) for doc in collection]

# Step 3: Create Inverted Index
def create_inverted_index(collection):
    inverted_index = {}
    for doc_id, doc in enumerate(collection):
        for term in doc.split():
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append(doc_id)
    return inverted_index

# Step 4: Vector Space Model
def build_tfidf_matrix(collection):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(collection)
    return vectorizer, tfidf_matrix

def retrieve_documents(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    return ranked_indices, similarities

# Step 5: Main Function
def main(file_path):
    print("Loading and preparing data...")
    collection = load_and_prepare_data(file_path)

    print("Preprocessing collection...")
    preprocessed_collection = preprocess_collection(collection)

    print("Creating inverted index...")
    inverted_index = create_inverted_index(preprocessed_collection)

    #print(inverted_index)
    
    print("Building TF-IDF matrix...")
    vectorizer, tfidf_matrix = build_tfidf_matrix(preprocessed_collection)

    print("System is ready! Enter your queries.")
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        preprocessed_query = preprocess_text(query)
        ranked_indices, similarities = retrieve_documents(preprocessed_query, vectorizer, tfidf_matrix)
        print("\nTop results:")
        for idx in ranked_indices[:5]:
            print(f"Doc {idx}: {collection[idx]} (Score: {similarities[idx]:.4f})\n")

# Execute the program
if __name__ == "__main__":
    file_path = "C://Adam//studies//Gl5//sem1//SRI//projet//FakePostings.csv"
    main(file_path)
