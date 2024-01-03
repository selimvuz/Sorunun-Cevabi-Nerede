from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import numpy as np

# Türkçe BERT modelini ve tokenizer'ı yükleyin
model_name = 'dbmdz/bert-base-turkish-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def get_bert_embedding(text, tokenizer, model):
    # Metni tokenize et ve PyTorch tensor'ına dönüştür
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Son katmanın gizli durumunu al ve düzleştir (flatten)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().flatten()


def split_document_into_segments(document_text, segment_length=512):
    # Belgeyi bölümlere ayırma
    tokenized_text = tokenizer.tokenize(document_text)
    return [tokenized_text[i:i + segment_length] for i in range(0, len(tokenized_text), segment_length)]


# JSON dosyalarını yükleyin
with open('Datasets/q&a_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('Datasets/extra_docs.json', 'r', encoding='utf-8') as file:
    extra_documents = json.load(file)

# Tüm dokümanları birleştirin
all_documents = data + extra_documents

# Dokümanları bölümlemek ve BERT temsillerine dönüştürmek
doc_segments_embeddings = []
for doc in all_documents:
    segments = split_document_into_segments(doc['document_text'])
    segments_embeddings = [get_bert_embedding(
        tokenizer.convert_tokens_to_string(seg), tokenizer, model) for seg in segments]
    doc_segments_embeddings.append(segments_embeddings)

# Soruları BERT temsillerine dönüştürün
question_embeddings = [get_bert_embedding(q['question_text'], tokenizer, model)
                       for doc in data for q in doc.get('questions', [])]

# Kosinüs benzerliği hesaplama ve tahminlerin saklanması
predictions = []
for question_emb in question_embeddings:
    max_similarity = -1
    most_similar_doc_index = -1
    most_similar_segment_index = -1
    for doc_index, segments in enumerate(doc_segments_embeddings):
        for segment_index, segment_emb in enumerate(segments):
            similarity = cosine_similarity([question_emb], [segment_emb])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_doc_index = doc_index
                most_similar_segment_index = segment_index
    predictions.append((most_similar_doc_index, most_similar_segment_index))

# Performans kriterlerini hesaplama
mrr = 0
precision = 0
correct_predictions = 0
total_questions = len(question_embeddings)  # Toplam soru sayısı

for i, (predicted_doc_index, predicted_segment_index) in enumerate(predictions):
    # Gerçek döküman ID'sini al
    question_index = i  # Soru indexi
    doc = all_documents[question_index //
                        len(data[0]['questions'])]  # İlgili dökümanı bul
    correct_doc_id = doc['document_id']

    # Tahmin edilen dökümanın ID'sini al
    predicted_doc_id = all_documents[predicted_doc_index]['document_id']

    # Doğruluk ve MRR hesaplama
    correct_predictions += int(correct_doc_id == predicted_doc_id)
    if correct_doc_id == predicted_doc_id:
        # MRR için sıralama (rank) hesaplama (Segment düzeyinde değil, döküman düzeyinde yapılır)
        rank = 1 / (question_index % len(data[0]['questions']) + 1)
        mrr += rank

precision = correct_predictions / total_questions
recall = correct_predictions / total_questions
mrr /= total_questions

# F1 skoru hesaplama
f1 = 2 * (precision * recall) / (precision +
                                 recall) if (precision + recall) > 0 else 0

# Güncellenmiş sonuçları yazdırma
print(f"Mean Reciprocal Rank (MRR): {mrr}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
