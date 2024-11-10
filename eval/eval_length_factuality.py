import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def length_percent_by_note(text_list,summary_list):
  num = len(text_list)
  avg = 0
  for i in range(num):
    note_l = len(text_list[i].split())
    summary_l = len(summary_list[i].split())
    p = summary_l/note_l
    avg += p
  avg = avg/len(text_list)
  return avg

def length_percent_by_all(text_list,summary_list):
  num = len(text_list)
  wc_note = 0
  wc_summary = 0
  for i in range(num):
    note_l = len(text_list[i].split())
    summary_l = len(summary_list[i].split())
    wc_note += note_l
    wc_summary += summary_l
  p = wc_summary/wc_note
  return p

def length_percent_by_adm(text_list,summary_list):
  adm_num = len(text_list)
  p_adm = 0
  for i in range(adm_num):
    note_l_adm = 0
    sum_l_adm = 0
    for note in text_list[i]:
      note_l_adm += len(note.split())
    for sum in summary_list[i]:
      sum_l_adm += len(sum.split())
    p_adm += sum_l_adm/note_l_adm
  p = p_adm/adm_num
  return p

def factuality(data):
    results = []
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
    factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2).to('cuda:0')

    for i in range(len(list(data.text))):
        summary = list(data.summary)[i]
        note = list(data.text)[i]
        input = [[summary, note]]
        tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True).to('cuda:0')
        result = torch.softmax(factkb(**tokens).logits, dim = 1)
        results.append(float(result[0][1]))
    return np.mean(np.array(results)), np.std(np.array(results))

