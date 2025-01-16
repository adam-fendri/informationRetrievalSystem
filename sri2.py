import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_prepare_data(true_path, fake_path, num_true=200, num_fake=20):
    # Load true data
    df_true = pd.read_csv(true_path)
    df_true = df_true.head(num_true)
    df_true['combined_text'] = df_true['title'] + " " + df_true['text']
    df_true['relevant'] = 1  # Mark as relevant
    
    # Load fake data
    df_fake = pd.read_csv(fake_path)
    df_fake = df_fake.head(num_fake)
    df_fake['combined_text'] = df_fake['title'] + " " + df_fake['text']
    df_fake['relevant'] = 0  # Mark as non-relevant
    
    # Combine datasets
    df_combined = pd.concat([df_true, df_fake], ignore_index=True)
    return df_combined['combined_text'].tolist(), df_combined['relevant'].tolist()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()
    tokens = text.split()  # Tokenize
    stopwords = set(open("stopwords.txt").read().split())
    tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords
    return " ".join(tokens)

def preprocess_collection(collection):
    return [preprocess_text(doc) for doc in collection]

def create_inverted_index(collection):
    inverted_index = {}
    for doc_id, doc in enumerate(collection):
        for term in doc.split():
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append(doc_id)
    return inverted_index

def build_tfidf_matrix(collection, method='ntc'):
    if method == 'ltc':
        # Using logarithmic term frequency
        vectorizer = TfidfVectorizer(sublinear_tf=True)
    else:
        # Using natural term frequency
        vectorizer = TfidfVectorizer(sublinear_tf=False)
    tfidf_matrix = vectorizer.fit_transform(collection)
    return vectorizer, tfidf_matrix


def retrieve_documents(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    return ranked_indices, similarities

def main(true_path, fake_path):
    print("Loading and preparing data...")
    collection, labels = load_and_prepare_data(true_path, fake_path)

    print("Preprocessing collection...")
    preprocessed_collection = preprocess_collection(collection)

    print("Creating inverted index...")
    inverted_index = create_inverted_index(preprocessed_collection)

    # Let user choose the retrieval method
    method = input("Choose the retrieval method (ntc or ltc): ").strip().lower()
    while method not in ['ntc', 'ltc']:
        print("Invalid input. Please choose 'ntc' or 'ltc'.")
        method = input("Choose the retrieval method (ntc or ltc): ").strip().lower()

    print("Building TF-IDF matrix...")
    vectorizer, tfidf_matrix = build_tfidf_matrix(preprocessed_collection, method=method)

    print("System is ready! Enter your queries.")
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        preprocessed_query = preprocess_text(query)
        ranked_indices, similarities = retrieve_documents(preprocessed_query, vectorizer, tfidf_matrix)

        top_indices = ranked_indices[:10]
        retrieved_labels = [labels[idx] for idx in top_indices]
        true_relevants = sum(retrieved_labels)

        precision = true_relevants / 10
        recall = true_relevants / sum([1 for label in labels[:len(retrieved_labels)] if label == 1]) if sum(labels) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        print("\nTop results:")
        for idx in top_indices:
            relevance = 'Relevant' if labels[idx] else 'Non-relevant'
            print(f"Doc {idx}: {collection[idx]} (Score: {similarities[idx]:.4f}, {relevance})")
        print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")

if __name__ == "__main__":
    true_path = "C://Adam//studies//Gl5//sem1//SRI//projet//True.csv"
    fake_path = "C://Adam//studies//Gl5//sem1//SRI//projet//Fake.csv"
    main(true_path, fake_path)
