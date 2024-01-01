import json

# JSON dosyasını açma ve içeriği okuma
with open('Dataset/soru-cevap.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('Dataset/ekstra-dokumanlar.json', 'r', encoding='utf-8') as file:
    extra = json.load(file)

document_ids = [doc['document_id'] for doc in data if 'document_id' in doc]
extra_ids = [doc['document_id'] for doc in extra if 'document_id' in doc]
question_ids = [doc['questions'] for doc in data if 'questions' in doc]

print(f"{len(document_ids)} adet döküman var.")
print(f"{len(extra_ids)} adet ekstra döküman var.")
print(f"{len(question_ids)*3} adet soru var.")
