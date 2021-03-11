import pickle
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def get_results(gold,pred):
  p,r,f1,_ = precision_recall_fscore_support(gold, pred, average='macro')
  correct, count = 0, 0
  for y_true, pred in zip(gold, pred):
      count += 1
      correct += (1 if y_true == pred else 0)
  acc = 100.0 * correct / count
  return [
      ('precision', p),
      ('recall', r),
      ('f1', f1),
      ('accuracy',acc),
  ]



label_map = ['Stay at Home, quarantine',
                               'Caution and advice to general public',
                               'Requesting for specific help, not comments',
                               'Reports/Complaints of utility services provider and landlords (inside the house); utlities policy and programs',
                               'COVID-19 Psychological impacts (need to have clear signals of mental and stessful impacts)',
                               'Everyday life inconvenience/disruption on edcation and life']
reverse_map = {v:k for k,v in enumerate(label_map)}
pred_file = sys.argv[1]
gold_file = sys.argv[2]
with open(pred_file,'rb') as f:
  data = pickle.load(f)
pred = [np.argmax(data[x]) for x in sorted(data)]
df = pd.read_csv(gold_file,sep='\t')
df['gold'] = df['Label'].apply(lambda x : reverse_map[x])
gold = df['gold'].to_list()
print(get_results(gold,pred))
