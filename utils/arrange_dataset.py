import os
import json

with open('dataset.json') as file_in:
	labels = json.load(file_in)


dct_test = {}
dct_train = {}

for vid_name in labels:

	if labels[vid_name]['set'] == 'test':
		dct_test[vid_name] = labels[vid_name]

	elif labels[vid_name]['set'] == 'train':
		dct_train[vid_name] = labels[vid_name]


with open('test.json', 'w') as file_out_test:
	json.dump(dct_test, file_out_test, indent=4)


with open('train.json', 'w') as file_out_train:
	json.dump(dct_train, file_out_train, indent=4)

