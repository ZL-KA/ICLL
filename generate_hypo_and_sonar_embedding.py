

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from mlsuperb2_whisper_sonar_lang_mapping import *
import os
import pandas as pd
from datasets import Dataset, load_dataset
import torch
from tqdm import tqdm
import json

import re # Python regex
import string # Python string manipulation library
import unicodedata # unicode punctuation detection
def remove_punctuation(sentence):
    '''https://multilingual.superbbenchmark.org/challenge-interspeech2025/challenge_overview#top'''
    new_sentence = ""
    for char in sentence:
        # all unicode punctuation is of type P
        if unicodedata.category(char).startswith('P'):
            continue
        else:
            new_sentence = f"{new_sentence}{char}"
    return new_sentence

def ml_superb_text_normalization(text, lang_code):    
    text = text.lower()
    text = text.replace(f'[{lang_code}]', '').strip()
    text = remove_punctuation(text)
    # remove space for Chinese/Japanese/Thai
    if lang_code in ['tha', 'jpn', 'cmn']:
        text = re.sub(r"\s", "", text)
    return text

def map_preprocess_mlsuperb_text(example, lang_code):
    example['transcript'] = ml_superb_text_normalization(example['transcript'], lang_code)
    return example


def do_whisper(dataset, file_path, do_asr=False, hypo_generation=False, whisper_lang_id=None, model_name="openai/whisper-large-v3", ):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")
    if whisper_lang_id!=None: # if requested AND supported, use it. Otherwise not.
        forced_ids = processor.get_decoder_prompt_ids(language=whisper_lang_id, task='transcribe')
    else:
        forced_ids = None  # let whisper auto-detect
    print(forced_ids)

    if do_asr:
        dataset = dataset.map(whisper_asr, fn_kwargs={'model':model, 'processor':processor, 'forced_ids':forced_ids})
        transcriptions=dataset['transcription']
        with open(file_path, "w", encoding="utf-8") as f:
            for item in transcriptions:
                f.write(item + "\n")

    if hypo_generation:

        dataset = dataset.map(whisper_asr_hypo, load_from_cache_file=False, fn_kwargs={'model':model, 'processor':processor, 'forced_ids':forced_ids})
        hypos=dataset['sample_hypos']
        with open(file_path.replace('.json', '_sample.json'), "w", encoding="utf-8") as f:
            json.dump(hypos, f, ensure_ascii=False, indent=2)
        hypos=dataset['beam_hypos']
        with open(file_path.replace('.json', '_beam.json'), "w", encoding="utf-8") as f:
            json.dump(hypos, f, ensure_ascii=False, indent=2)
    print(f'Saved {file_path}')


def whisper_asr(example, model, processor, forced_ids):
    inputs = processor(example["audio"]["array"], return_tensors="pt", sampling_rate=example["audio"]["sampling_rate"])
    input_features = inputs.input_features
    generated_ids = model.generate(inputs=input_features.to(model.device), forced_decoder_ids=forced_ids)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    example['transcription']=transcription
    return example
    
def whisper_asr_hypo(example, model, processor, forced_ids):
    inputs = processor(example["audio"]["array"], return_tensors="pt", sampling_rate=example["audio"]["sampling_rate"])
    input_features = inputs.input_features
    generated_ids = model.generate(inputs=input_features.to(model.device), num_beams=5, num_return_sequences=5, forced_decoder_ids=forced_ids) # This brings little difference
    decoded = [processor.tokenizer.decode(out, skip_special_tokens=True) for out in generated_ids]
    example['beam_hypos']= decoded
    
    generated_ids = model.generate(inputs=input_features.to(model.device), do_sample=True, top_k=50, top_p=0.95, num_return_sequences=5, temperature=0.9, forced_decoder_ids=forced_ids)
    decoded = [processor.tokenizer.decode(out, skip_special_tokens=True) for out in generated_ids]
    example['sample_hypos']= decoded
    return example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang='ml_superb2'
lang_sonar_mapping = mapping_no_empty
dataset= load_dataset('espnet/ml_superb_hf')
dataset = dataset.rename_columns({'text': 'transcript'})
dataset_test = dataset['dev']
dataset_train=dataset['train']

t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                            tokenizer="text_sonar_basic_encoder",
                                            device=device)

lang_code_list = sorted(list(set(dataset_test['language'])))

do_whisper_with_langid=True # Test using lang id or not in whisper
do_asr=False
do_hypos_generation=True # whether to generate hypos for ASR

for idx, lang_code in enumerate(lang_code_list):
    print(f'Processing {idx} language: {lang_code}')
    the_dataset_train=dataset_train.filter(lambda example: example['language'] == lang_code)
    the_dataset_test= dataset_test.filter(lambda example: example['language'] == lang_code)
    print('the_dataset_train: ', the_dataset_train)
    print('the_dataset_test: ', the_dataset_test)
    # preprocess transcript
    the_dataset_train = the_dataset_train.map(map_preprocess_mlsuperb_text, fn_kwargs={'lang_code': lang_code})
    the_dataset_test = the_dataset_test.map(map_preprocess_mlsuperb_text, fn_kwargs={'lang_code': lang_code})

    sonar_lang = lang_sonar_mapping[lang_code][0]
    
    # Load ASR results 
    asr_results_path=f'/TBD/{lang_code}.txt'
    sonar_results_path=f'/TBD/{lang_code}.json'
    if do_hypos_generation:
        asr_results_path=asr_results_path.replace('whisper_results', 'whisper_hypos').replace('.txt', '.json')
        sonar_results_path=sonar_results_path.replace('sonar_results', 'sonar_hypos')
    
    if do_whisper_with_langid:
        whisper_lang_id = mlsuperb2_to_whisper[lang_code] if mlsuperb2_to_whisper[lang_code] != "not_available" else None
        print(f'whisper_lang_id: {whisper_lang_id}')
        asr_results_path=asr_results_path.replace(f'/{lang_code}', f'_with_langid/{lang_code}')
        sonar_results_path=sonar_results_path.replace(f'/{lang_code}', f'_with_langid/{lang_code}')
    if not os.path.exists(asr_results_path):
        do_whisper(the_dataset_test, asr_results_path, do_asr=do_asr, hypo_generation=do_hypos_generation, whisper_lang_id=whisper_lang_id)
    
    
    
    
    if not os.path.exists(sonar_results_path):
        sonar_results=[]
        if len(the_dataset_train)!=0:
            results = [line.strip() for line in open(asr_results_path, encoding="utf-8")]
            for result in tqdm(results):
                sentences=the_dataset_train['transcript']
                embeddings = t2vec_model.predict(sentences, source_lang=sonar_lang)
                sonar_results.append([torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[idx].unsqueeze(0)).item() for idx in range(1, embeddings.shape[0])])
        # import pdb;pdb.set_trace()
        with open(sonar_results_path, 'w') as f:
                json.dump(sonar_results, f)

