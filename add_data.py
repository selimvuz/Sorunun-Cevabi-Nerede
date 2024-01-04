import os
import json

# The directory where the text files are located
text_files_directory = 'Datasets/Documents/'
# The JSON file to update
json_file_path = 'Datasets/extra_docs.json'

# Function to read text files and return their content
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to update the JSON data with the text files content
def update_json_with_text_files(start_id, end_id, text_files_dir, json_path):
    new_data = []

    # Iterate through the specified range of file IDs
    for file_id in range(start_id, end_id + 1):
        file_name = f"{file_id}.txt"
        file_path = os.path.join(text_files_dir, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            # Read the content of the file
            document_text = read_text_file(file_path)
            # Append the data in the required format
            new_data.append({
                "document_id": file_id,
                "document_text": document_text
            })

    # Load the existing JSON data
    with open(json_path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)

    # Append the new data to the existing data
    updated_data = existing_data + new_data

    # Write the updated data back to the JSON file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(updated_data, json_file, ensure_ascii=False, indent=4)


# Update the JSON file with the new documents
update_json_with_text_files(401, 500, text_files_directory, json_file_path)
