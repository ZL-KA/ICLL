import evaluate
cer_metric=evaluate.load('cer')
wer_metric=evaluate.load('wer')

import math
import random
import numpy as np
import os
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything()

from datasets import Dataset, load_dataset, Audio
from transformers import AutoTokenizer, AutoModelForCausalLM


dataset = None


# Load hypothesis with scores
hypothesis_path = f'/TBD/{lang}_wav2vec2_10best.json'

import json
with open(hypothesis_path, "r", encoding="utf-8") as f:
    hypo_ds = json.load(f)

hypo_asr_scores = [item['logit_score'] for item in hypo_ds]
hypo_text = [item['text'] for item in hypo_ds]

dataset_test = dataset['test']
dataset_test = dataset_test.add_column('10_best', hypo_text)
dataset_test = dataset_test.add_column('10_best_asr_scores', hypo_asr_scores)

print(f'*************** \n {lang} \n **********************')




def oracel_selection_from10best(example, wer_metric):
    hypotheses = example['10_best']
    reference=example['transcript']
    wer_metric = [wer_metric.compute(predictions=[hypothesis], references=[reference]) for hypothesis in hypotheses]
    min_wer_index = wer_metric.index(min(wer_metric))
    best_hypothesis = hypotheses[min_wer_index]
    example['best_selection_asr_oracel']= best_hypothesis
    return example


    
def do_icl_llama_selection(example, idx, strategy, training_transcripts, num_sample, model, tokenizer):
    hypotheses = example['10_best']
    indices=select_indices(strategy, len(training_transcripts), num_sample, hypo_results=hypo_results, oracle_results=oracle_results, speech_results=speech_results, key_idx=idx)
    selected_samples = [training_transcripts[idx] for idx in indices]
    prompt = '\n'.join(selected_samples) + '\n'
    hypo_ppls=[]
    for hypothesis in hypotheses:
        with torch.no_grad():
            inputs = tokenizer(prompt+hypothesis, return_tensors='pt')
            outputs = model(**inputs.to(model.device))
            valid_logits_start = len(tokenizer(prompt).input_ids)
            valid_logits = outputs.logits[:, valid_logits_start:, :]
            # Cross_entropy
            loss = model.loss_function(valid_logits, labels=inputs.input_ids[:, valid_logits_start:], vocab_size=model.config.vocab_size)
            hypo_ppls.append(math.exp(loss))
    try:
        normalize_acoustic = normalize_to_unit_range(example['10_best_asr_scores'])
    except:
        normalize_acoustic = [0]*len(normalize_ppls)
    normalize_ppls = normalize_to_unit_range(hypo_ppls)
    combined_scores = [a + l for a, l in zip(normalize_acoustic, normalize_ppls)]
    best_hypo_index = combined_scores.index(max(combined_scores))
    example['hypo_select']=hypotheses[best_hypo_index]
    return example


def normalize_to_unit_range(values):
    """
    Normalize a list of floats to [0, 1] according to rules:
      - All positive: min -> 1, max -> 0 (inverted scale)
      - All negative: max -> 1, min -> 0
      - Mixed signs: raises ValueError
    """
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmin < 0 and vmax > 0:
        raise ValueError("Values contain both positive and negative numbers.")
    if vmin == vmax:
        return [1.0 for _ in values]  # everything identical
    if vmax <= 0:
        # all negatives: max → 1, min → 0
        return [(x - vmin) / (vmax - vmin) for x in values]
    else:
        # all positives: min → 1, max → 0 (invert)
        return [1 - (x - vmin) / (vmax - vmin) for x in values]



# Run selection

# Oracel
dataset_test = dataset_test.map(oracel_selection_from10best, fn_kwargs={'wer_metric': wer_metric})




# ICL_llama
strategy='sample_best3'
hypo_results=None
oracle_results=None
speech_results=None
if strategy in ['corpus_best1', 'corpus_best3','sample_best1','sample_best3', 'sample_audio_text_best1']:
    file_path=f'TBD/{lang}_sonar_cossimi_hypos_results.json'
    with open(file_path,'r') as f:
        # results is one list of [num_test lists] of [num_hypo lists] of [number_train_sample values]
        hypo_results=json.load(f)
    assert len(hypo_results[0][0])==len(dataset['train'])
if strategy in ['sample_oracle']:
    file_path=f'TBD/{lang}_sonar_cossimi_text_oracle_results.json'
    with open(file_path,'r') as f:
        # results is one list of [num_test lists] of [num_hypo lists] of [number_train_sample values]
        oracle_results=json.load(f)
    assert len(oracle_results[0])==len(dataset['train'])

def select_indices(strategy, total_sample_length, num_sample, hypo_results=None, 
                       oracle_results=None, speech_results=None, key_idx=None):
        if strategy=='corpus_random':
            indices = random.sample(range(total_sample_length), num_sample)
            
        elif strategy =='sample_oracle':
            sonar_scores=oracle_results[key_idx]
            indices= np.argsort(sonar_scores)[-num_sample:][::-1].tolist()
            
        elif strategy =='sample_best1':
            sample_hypo_result_best=hypo_results[key_idx][0]
            indices=np.argsort(sample_hypo_result_best)[-num_sample:][::-1].tolist()
            
        elif strategy=='sample_best3':
            sample_hypos_result_best=hypo_results[key_idx]
            first_3_hypo_collect = []
            for item in sample_hypos_result_best:
                max_idx = min(len(item), 3) # Could be less than 3 hypo
                collect_items = []
                for idx in range(max_idx):
                    collect_items.append(item)
                summed_item = np.sum(np.array(collect_items), axis=0).tolist()
                first_3_hypo_collect.append(summed_item)
            sample_best3=np.sum(np.array(first_3_hypo_collect), axis=0).tolist()
            indices=np.argsort(sample_best3)[-num_sample:][::-1].tolist()
        else:
            import pdb
            pdb.set_trace()
        return indices



num_samples = [50]


for num_sample in num_samples:
    save_path = None
    selection_pool = dataset['train']['transcript']
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    dataset_test_icl_llama = dataset_test.map(do_icl_llama_selection, with_indices=True, 
                                              fn_kwargs={'strategy':strategy, 'training_transcripts':dataset['train']['transcript'],
                                                         'num_sample':num_sample, 'model':model, 'tokenizer':tokenizer})
    
    items_to_remove=['transcript', 'audio', 'speaker_id', '10_best', '10_best_asr_scores']
    items_to_score=[item for item in dataset_test_icl_llama.column_names if item not in items_to_remove]
    scores = []
    for item in items_to_score:
        score = wer_metric.compute(predictions=dataset_test_icl_llama[item], references=dataset_test_icl_llama['transcript'])
        score = round(score*100, 2)
        scores.append(score)
    print(items_to_score)
    print(scores)
    
    


