import argparse
import json
from transformers import MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

def translate_text(text, model, tokenizer, device, beam_size):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        translated = model.generate(**inputs, num_beams=beam_size, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        print(translated_text)
    return translated_text

def translate_file(input_file_path, output_file_path, model, tokenizer, device, beam_size, keys_to_translate):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if keys_to_translate == ['Statement']:  # Special handling for top-level JSON files
        for record_id in data:
            if 'Statement' in data[record_id]:
                data[record_id]['Statement'] = translate_text(data[record_id]['Statement'], model, tokenizer, device, beam_size)
    else:  # General handling for nested JSON files
        for key in keys_to_translate:
            if key in data:
                data[key] = [translate_text(text, model, tokenizer, device, beam_size) for text in data[key]]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_folder(folder_path, output_folder, model, tokenizer, device, beam_size, is_root=True):
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists
    for entry in os.listdir(folder_path):
        input_path = os.path.join(folder_path, entry)
        output_path = os.path.join(output_folder, entry)
        if os.path.isdir(input_path):
            # Recursive call for subdirectories
            process_folder(input_path, output_path, model, tokenizer, device, beam_size, is_root=False)
        elif input_path.endswith('.json'):
            # Process JSON files
            keys_to_translate = ['Statement'] if is_root else ['Intervention', 'Eligibility', 'Results', 'Adverse Events']
            translate_file(input_path, output_path, model, tokenizer, device, beam_size, keys_to_translate)

def main():
    parser = argparse.ArgumentParser(description="Translate JSON files using NMT models.")
    parser.add_argument('--model_name_or_path', type=str, default=5, help='model name or path')
    parser.add_argument('--data', type=str, required=True, help='Input folder containing JSON files')
    parser.add_argument('--output', type=str, required=True, help='Output folder for translated JSON files')
    parser.add_argument('--src_lang', type=str, default='en', help='Source language code')
    parser.add_argument('--tgt_lang', type=str, default='zh', help='Target language code')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for translation')

    args = parser.parse_args()

    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(args.device)  # or is it AutoModelForConditionalGeneration ???

    process_folder(args.data, args.output, model, tokenizer, args.device, args.beam_size)

    print("Translation completed for all files!")

if __name__ == "__main__":
    main()