from transformers import GPT2Model, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json

print("Model yükleniyor...")

# mGPT modelini ve tokenizer'ı yükleyin
model_name = 'ai-forever/mGPT'  # Model ismini değiştirin
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

i = 1


def get_gpt2_embedding(text, tokenizer, model):
    # Metni tokenize et ve PyTorch tensor'ına dönüştür
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Son katmanın gizli durumunu al ve düzleştir (flatten)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().flatten()


def split_document_into_segments(document_text, segment_length=48):
    # Belgeyi bölümlere ayırma
    tokenized_text = tokenizer.tokenize(document_text)
    return [tokenized_text[i:i + segment_length] for i in range(0, len(tokenized_text), segment_length)]


print("Veriler yükleniyor...")

# JSON dosyalarını yükleyin
with open('Datasets/q&a_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('Datasets/extra_docs.json', 'r', encoding='utf-8') as file:
    extra_documents = json.load(file)

# Tüm dokümanları birleştirin
all_documents = data + extra_documents

print("Metinler vektör temsillerine dönüştürülüyor...")

# Dokümanları bölümlemek ve GPT-2 temsillerine dönüştürmek
doc_segments_embeddings = []
for doc in all_documents:
    segments = split_document_into_segments(doc['document_text'])
    segments_embeddings = [get_gpt2_embedding(
        tokenizer.convert_tokens_to_string(seg), tokenizer, model) for seg in segments]
    doc_segments_embeddings.append(segments_embeddings)
    print(f"Hesaplanıyor: {i}/{len(all_documents)}")
    i += 1

# Soruları GPT-2 temsillerine dönüştürün
question_embeddings = [get_gpt2_embedding(q['question_text'], tokenizer, model)
                       for doc in data for q in doc.get('questions', [])]

print("Kosinüs benzerliği hesaplanıyor...")

# Kosinüs benzerliği hesaplama ve sıralı tahminlerin saklanması
predictions = []
for question_emb in question_embeddings:
    similarities = []
    for doc_index, segments in enumerate(doc_segments_embeddings):
        for segment_index, segment_emb in enumerate(segments):
            similarity = cosine_similarity([question_emb], [segment_emb])[0][0]
            similarities.append((similarity, doc_index, segment_index))

    # Benzerliklere göre sırala ve en yüksek skorluları seç
    sorted_similarities = sorted(
        similarities, key=lambda x: x[0], reverse=True)
    predictions.append(sorted_similarities)

print("Performans kriterleri hesaplanıyor...")

# Performans kriterlerini hesaplama
mrr = 0
precision = 0
correct_predictions = 0
total_questions = len(question_embeddings)  # Toplam soru sayısı

# Performans kriterlerini hesaplama
mrr = 0
precision = 0
correct_predictions = 0
total_questions = len(question_embeddings)  # Toplam soru sayısı

# Precision için yalnızca en yüksek skorlu tahminleri değerlendir
for i, sorted_sims in enumerate(predictions):
    question_index = i
    doc = all_documents[question_index // len(data[0]['questions'])]
    correct_doc_id = doc['document_id']

    # En yüksek skorlu tahmin
    predicted_doc_index = sorted_sims[0][1]
    predicted_doc_id = all_documents[predicted_doc_index]['document_id']

    # Precision hesaplama
    if correct_doc_id == predicted_doc_id:
        correct_predictions += 1

# MRR için tüm tahminlerin sıralamasını dikkate al
mrr = 0
for i, sorted_sims in enumerate(predictions):
    question_index = i
    doc = all_documents[question_index // len(data[0]['questions'])]
    correct_doc_id = doc['document_id']

    # MRR hesaplama
    for rank, (similarity, doc_index, _) in enumerate(sorted_sims, start=1):
        predicted_doc_id = all_documents[doc_index]['document_id']
        if correct_doc_id == predicted_doc_id:
            mrr += 1 / rank
            break

precision = correct_predictions / total_questions
mrr /= total_questions

print("Sonuçlar:\n")
print(f"Mean Reciprocal Rank (MRR): {mrr}")
precision_percentage = precision * 100
print(f"Precision: %{precision_percentage}")
