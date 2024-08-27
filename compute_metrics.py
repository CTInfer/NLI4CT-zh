import evaluate
from evaluate import load
import pandas as pd
import numpy as np

ref_file = pd.read_csv('gold_references.csv')
ref_file = ref_file.dropna(how='all') 

predictions = [i for i in ref_file['m2m-100']]

references = [[i] for i in ref_file['zh_corrected']]

sacrebleu = evaluate.load("sacrebleu")
results = sacrebleu.compute(predictions=predictions, references=references, tokenize='zh')

print("Sacrebleu", results)


ter = evaluate.load("ter")
results_ter = ter.compute(predictions=predictions, references=references, ignore_punct=True, support_zh_ja_chars=True)

print("TER", results_ter)


chrf = evaluate.load("chrf")
results_chrf = chrf.compute(predictions=predictions, references=references)
print("CHRF", results_chrf)

bertscore = load("bertscore")
references_bs = [i for i in ref_file['zh_corrected']]
bertscore_score = bertscore.compute(predictions=predictions, references=references_bs, lang="zh")
print(bertscore_score)
P = np.mean(bertscore_score["precision"])
R = np.mean(bertscore_score["recall"])
F1 = np.mean(bertscore_score["f1"])
print(f"BERTScore Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")
