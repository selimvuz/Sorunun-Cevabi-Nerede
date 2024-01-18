from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
import json
import numpy as np

print("Modeller yükleniyor...")

# BERT ve mGPT modellerini ve tokenizer'ları yükleyin
bert_model_name = 'dbmdz/bert-base-turkish-uncased'
gpt_model_name = 'ai-forever/mGPT'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPT2Model.from_pretrained(gpt_model_name)

i = 1


def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().flatten()


def get_gpt2_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().flatten()


def split_document_into_segments(document_text, tokenizer, segment_length=48):
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

print("Metinler vektör temsillerine dönüştürülüyor... Bu uzun sürebilir!")

# Dokümanları ve soruları bölümlemek, gömülere dönüştürmek
bert_doc_embeddings = []
gpt_doc_embeddings = []
for doc in all_documents:
    bert_segments = split_document_into_segments(
        doc['document_text'], bert_tokenizer)
    gpt_segments = split_document_into_segments(
        doc['document_text'], gpt_tokenizer)

    bert_segments_embeddings = [get_bert_embedding(bert_tokenizer.convert_tokens_to_string(
        seg), bert_tokenizer, bert_model) for seg in bert_segments]
    gpt_segments_embeddings = [get_gpt2_embedding(gpt_tokenizer.convert_tokens_to_string(
        seg), gpt_tokenizer, gpt_model) for seg in gpt_segments]

    bert_doc_embeddings.append(bert_segments_embeddings)
    gpt_doc_embeddings.append(gpt_segments_embeddings)
    print(f"Hesaplanıyor: {i}/{len(all_documents)}")
    i += 1

# Soruları temsilcilerine dönüştürmek
bert_question_embeddings = []
gpt_question_embeddings = []
for doc in data:
    for q in doc.get('questions', []):
        bert_q_emb = get_bert_embedding(
            q['question_text'], bert_tokenizer, bert_model)
        gpt_q_emb = get_gpt2_embedding(
            q['question_text'], gpt_tokenizer, gpt_model)

        bert_question_embeddings.append(bert_q_emb)
        gpt_question_embeddings.append(gpt_q_emb)

print("Kosinüs benzerliği hesaplanıyor...")

# Kosinüs benzerliği hesaplama ve sıralı tahminlerin saklanması
predictions = []
for bert_q_emb, gpt_q_emb in zip(bert_question_embeddings, gpt_question_embeddings):
    similarities = []
    for doc_index, (bert_segments, gpt_segments) in enumerate(zip(bert_doc_embeddings, gpt_doc_embeddings)):
        for segment_index, (bert_emb, gpt_emb) in enumerate(zip(bert_segments, gpt_segments)):
            bert_similarity = cosine_similarity([bert_q_emb], [bert_emb])[0][0]
            gpt_similarity = cosine_similarity([gpt_q_emb], [gpt_emb])[0][0]
            combined_similarity = bert_similarity + gpt_similarity
            similarities.append(
                (combined_similarity, doc_index, segment_index))

    # Benzerliklere göre sırala ve en yüksek skorluları seç
    sorted_similarities = sorted(
        similarities, key=lambda x: x[0], reverse=True)
    predictions.append(sorted_similarities)

print("Performans kriterleri hesaplanıyor...")

# Performans kriterlerini hesaplama
correct_predictions = 0
mrr = 0
total_questions = len(bert_question_embeddings)
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

    # Precision hesaplama
    predicted_doc_index = sorted_sims[0][1]
    predicted_doc_id = all_documents[predicted_doc_index]['document_id']
    if correct_doc_id == predicted_doc_id:
        correct_predictions += 1

precision = correct_predictions / total_questions
mrr /= total_questions

print("Sonuçlar:\n")
print(f"Mean Reciprocal Rank (MRR): {mrr}")
precision_percentage = precision * 100
print(f"Precision: %{precision_percentage}")
