from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# JSON dosyalarını yükleyin
with open('Datasets/q&a_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('Datasets/extra_docs.json', 'r', encoding='utf-8') as file:
    extra_documents = json.load(file)

# Tüm dokümanları birleştirin
all_documents = data + extra_documents

# Dokümanları ve soruları hazırlayın
documents = [doc['document_text'].lower() for doc in all_documents]
questions = [q['question_text'].lower()
             for doc in data for q in doc.get('questions', [])]

# TF-IDF Vektörleştirici oluşturma ve doküman vektörleri hesaplama
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# Doküman ID'lerini ve soruları eşleştirme
document_ids = {doc['document_id']: i for i, doc in enumerate(all_documents)}
question_to_doc_id = {f"{doc['document_id']}_{q['question_id']}": doc['document_id']
                      for doc in data for q in doc.get('questions', [])}

# Kosinüs benzerliği hesaplama ve tahminlerin saklanması
predictions = []
for doc in data:
    doc_id = doc['document_id']
    for q in doc.get('questions', []):
        question_id = f"{doc_id}_{q['question_id']}"  # Bu satırı ekleyin
        question_vector = vectorizer.transform([q['question_text'].lower()])
        cosine_similarities = cosine_similarity(question_vector, doc_vectors)
        most_similar_document_index = cosine_similarities.argmax()
        predicted_doc_id = all_documents[most_similar_document_index]['document_id']
        predictions.append((question_id, predicted_doc_id))

# Doğruluk ölçütlerini hesaplama
mrr = 0
precision = 0
correct_predictions = 0
for question_id, predicted_doc_id in predictions:
    correct_doc_id = question_to_doc_id[question_id]
    correct_index = document_ids[correct_doc_id]
    predicted_index = document_ids[predicted_doc_id]
    rank = 1 / (predicted_index + 1)
    mrr += rank
    correct_predictions += int(correct_doc_id == predicted_doc_id)

precision = correct_predictions / len(predictions)

mrr /= len(questions)

# Güncellenmiş sonuçları yazdırma
print(f"Mean Reciprocal Rank (MRR): {mrr}")
print(f"Precision: {precision}")
