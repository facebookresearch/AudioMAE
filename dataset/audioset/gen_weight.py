# -*- coding: utf-8 -*-
# modified from Yuan Gong@MIT's gen_weight_file

import argparse
import json
import numpy as np
import sys, os, csv

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def gen_weight(josn_file, label_file, output_file):
    index_dict = make_index_dict(label_file)
    label_count = np.zeros(527)

    with open(josn_file, 'r', encoding='utf8')as fp:
        data = json.load(fp)['data']

    for sample in data:
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            label_count[label_idx] = label_count[label_idx] + 1

    label_weight = 1000.0 / (label_count + 0.01)
    #label_weight = 1000.0 / (label_count + 100)

    sample_weight = np.zeros(len(data))
    for i, sample in enumerate(data):
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            # summing up the weight of all appeared classes in the sample, note audioset is multiple-label classification
            sample_weight[i] += label_weight[label_idx]
    np.savetxt(output_file, sample_weight, delimiter=',')
    print(label_weight)


if __name__ == '__main__':
    #args = parser.parse_args()
    json_file='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all.json'
    label_file='/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv'
    output_file='./weight_train_all.csv'
    gen_weight(json_file,label_file,output_file)


