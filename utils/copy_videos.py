import os
import json
import shutil



with open('test.json') as f:
	test_labels = json.load(f)


copy_path = 'test'

for i, vid_name in enumerate(test_labels):

	back_slash = vid_name.rfind('/')
	saved_file_name = vid_name[back_slash + 1:]

	shutil.copy2(vid_name, copy_path)

	if i % 100 == 0:
		print(vid_name)













