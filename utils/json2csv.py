import os
import json

with open('test.json') as file_in:
	labels = json.load(file_in)



with open('GT.csv', 'w') as fout:
	
	for vid in labels:
		ind = vid.rfind('/')
		file_name = vid[ind + 1 :]
		label_vid = 1 if labels[vid]['label'] == 'fake' else 0
		fout.write(f'{file_name},{label_vid}\n')



			





