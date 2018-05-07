import csv
import numpy as np

class VocabDict(object):
  def __init__(self):
    self.dict = {'<unk>': 0}

  def fit(self, word):
    if word not in self.dict:
      self.dict[word] = len(self.dict)

  def size(self):
    return len(self.dict)

  def transform(self, word):
    if word in self.dict:
      return self.dict[word]
    return 0

  def fit_and_transform(self, word):
    self.fit(word)
    return self.transform(word)

def to_categorical(y, target_dict, mode_transform=False):
  result = []
  tmp_y = []
  for y_i in y:
    for y_j in y_i:
      tmp_y.append(y_j)
  if mode_transform == False:
    l = len(np.unique(tmp_y)) + 1
  else:
    l = target_dict.size()
  for i, d in enumerate(y):
    tmp = [0.] * l
    for _i, _d in enumerate(d):
      if mode_transform == False:
        tmp[target_dict.fit_and_transform(_d)] = 1.
      else:
        tmp[target_dict.transform(_d)] = 1.
    result.append(tmp)
  return result

def load_csv(filepath, target_columns=-1, columns_to_ignore=None,
    has_header=True, n_classes=None, target_dict=None, mode_transform=False):

  if isinstance(target_columns, list) and len(target_columns) < 1:
    raise Exception('target_columns must be list with one value at least')

  from tensorflow.python.platform import gfile
  with gfile.Open(filepath) as csv_file:
    data_file = csv.reader(csv_file)
    if not columns_to_ignore:
      columns_to_ignore = []
    if has_header:
      header = next(data_file)

    data, target = [], []
    for i, d in enumerate(data_file):
      data.append([_d for _i, _d in enumerate(d) if _i not in target_columns and _i not in columns_to_ignore])
      target.append([_d for _i, _d in enumerate(d) if _i in target_columns])
    if target_dict is None:
      target_dict = VocabDict()
    target = to_categorical(target, target_dict=target_dict, mode_transform=mode_transform)
    return data, target

def load_txt(filepath, target_dict, mode_transform=False):
  f = open(filepath)
  data, target = [], []
  for line in f.readlines():
    doc, tags = line.strip().split("\t")
    data.append([doc])
    target.append(tags.split(","))
  f.close()
  if target_dict is None:
    target_dict = VocabDict()
  target = to_categorical(target, target_dict=target_dict, mode_transform=mode_transform)
  return data, target

