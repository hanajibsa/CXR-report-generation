import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import json
from rouge import Rouge
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu

from nltk.tokenize import word_tokenize

total_js = 0
total_rouge_n = 0
total_rouge_l = 0
total_meteor = 0
total_bleu_score = 0


def jaccard_similarity(text1, text2):
    # Tokenize the texts
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate the Jaccard similarity coefficient
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    similarity = intersection / union

    return similarity


# List of generated texts
golden = "gold/gold_pubmed_80_1000.json"
# f = open(golden)

golden_list = []
with open(golden, 'r') as f:
    line = f.readline()
    while line:
        data = json.loads(line)
        line = f.readline()
        # line = line.strip()
        # line = line[1:-1]
        golden_list.append(line)

# gold_data = json.load(f)

pred = "pred/pred_pubmed_80_1000.json"
pred_list = []
with open(pred, 'r') as f:
    line = f.readline()
    while line:
        data = json.loads(line)
        line = f.readline()
        # line = line.strip()
        line1 = line[1:-2] + "\n"
        pred_list.append(line1)

for x in range(len(golden_list)):
    text1 = golden_list[x]
    text2 = pred_list[x]

    if text1 == '"' or text2 == '\n':
        continue
    js = jaccard_similarity(text1, text2)
    total_js = total_js + js

    # Create a Rouge object with the desired metrics and options
    rouge = Rouge(metrics=['rouge-2', 'rouge-l'])

    # Calculate the ROUGE scores
    scores = rouge.get_scores(text2, text1)
    total_rouge_n = total_rouge_n + scores[0]['rouge-2']['f']
    total_rouge_l = total_rouge_l + scores[0]['rouge-l']['f']

    # Calculate meteor score
    reference = word_tokenize(text1)
    hypothesis = word_tokenize(text2)
    meteor_scores = meteor_score.meteor_score([reference], hypothesis)
    total_meteor = total_meteor + meteor_scores

    # Calculate the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis)
    total_bleu_score = total_bleu_score + bleu_score

average_js = total_js / len(golden_list)
average_rouge_n = total_rouge_n / len(golden_list)
average_rouge_l = total_rouge_l / len(golden_list)
average_meteor = total_meteor / len(golden_list)
average_bleu_Score = total_bleu_score/ len(golden_list)

print('average_js: ', average_js)
print('average_rouge_n: ', average_rouge_n)
print('average_rouge_l: ', average_rouge_l)
print('average_meteor: ', average_meteor)
print('average_bleu_Score: ', average_bleu_Score)

# Tokenize generated texts and reference texts
#tokenized_generated_texts = [nltk.word_tokenize(text) for text in generated_texts]
#tokenized_reference_texts = [[nltk.word_tokenize(text) for text in refs] for refs in reference_texts]

# Calculate BLEU score
#bleu_score = corpus_bleu(tokenized_reference_texts, tokenized_generated_texts)

print("BLEU score:", bleu_score)