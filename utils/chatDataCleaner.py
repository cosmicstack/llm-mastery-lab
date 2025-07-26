import os
import re
import zipfile
from pathlib import Path

ZIP_FILES_PATH = Path(__file__).parent.parent / "data/wa_zips/"
OUTPUT_FILE = Path(__file__).parent.parent / "data/cleaned_chat_data.txt"

def process_zip_data():
    output_data = []
    zip_files = Path(ZIP_FILES_PATH).glob("*.zip")

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as z:
            print(f"Processing {zip_file}...")
            for file_info in z.infolist():
                if file_info.filename.endswith('.txt'):
                    print(f"Reading {file_info.filename}...")
                    with z.open(file_info) as f:
                        for line in f.read().decode('utf-8').split('\n'):
                            if 'K:' in line:
                                k_index = line.find("K:")
                                if k_index != -1:
                                    message = line[k_index + 2:].strip()
                                    output_data.append(message)
    
    if OUTPUT_FILE.exists():
        # delete the existing file
        OUTPUT_FILE.unlink()

    with open(OUTPUT_FILE, 'w') as out_file:
        for item in output_data:
            out_file.write(f"{item}\n")

def clean_data(output_file=OUTPUT_FILE):
    if not output_file.exists():
        process_zip_data()
    
    with open(output_file, 'r') as file:
        lines = file.readlines()

        cleaned_lines = []
        
        #TODO: There are some issues with the regex, need to fix them

        for line in lines:
            # Remove URLs
            line = re.sub(r'http[s]?://\S+', '', line)
            # Remove Emails
            line = re.sub(r'\b[A-Za-z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '', line)
            # Remove @ mentions
            line = re.sub(r'@[\w.-]+', '', line)
            cleaned_lines.append(line)
        
        # Remove empty lines
        cleaned_lines = [line for line in cleaned_lines if line.strip()]
        cleaned_lines = [line for line in lines if '\u200E' not in line]

    with open(output_file, 'w') as file:
        file.writelines(cleaned_lines)

if __name__ == "__main__":
    clean_data()
    print(f"Data cleaned and saved to {OUTPUT_FILE}")