# Running the Assessment Framework
1. Clone this repo first
2. Refer to run.py for examples of use of statistics class.


# Re-generating results for DFDC
## Selim Seferbekov
 
1. Clone https://github.com/selimsef/dfdc_deepfake_challenge repo in a local directory.
2. Build docker image using the DockerFile in that repo.

`   docker build -t df .   `

3. Run the Docker image by mounting pre-trained weight files and test data.

`   docker run --runtime=nvidia --ipc=host --rm --volume <current_directory>:/code --volume <test_videos_directory>:/test -it <docker_image> /bin/bash   `

4. Run predict_folder.py with correct path to weights.

`  python predict_folder.py \
 --test-dir "/test" \
 --models final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36 \
  final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19 \
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29 \
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31 \
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37 \
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40 \
  final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23   `

5. Transfer submission.csv file out of the container with docker cp command.

`   docker cp <container_name>:/code/submission.csv submission_sef.csv

## NtechLab

1. Clone https://github.com/NTech-Lab/deepfake-detection-challenge.git repo in a local directory.
2. Build docker image using the DockerFile in that repo.
3. Run the Docker image by mounting pre-trained weight files and test data.
4. Run predict.py with correct path to weights.
5. Transfer submission.csv file out of the container with docker cp command.

## TeamVM

1. Clone https://github.com/cuihaoleo/kaggle-dfdc.git repo in a local directory.
2. load packages given in requirements.txt file
4. Run submission.py with correct path to weights and path to test data


# Generating a Dataset to train DeepFake Detectors
Following script generates aligned and cropped faces from the videos in a source folder and saves these images in the target folder.

1. Clone https://github.com/cuihaoleo/kaggle-dfdc.git repo in a local directory.
2. load packages given in requirements.txt file
3. Run shell script make_dataset.sh by modifying source and target folders as you need.





