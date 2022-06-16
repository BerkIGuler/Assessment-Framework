# Running the Assessment Framework
1. Clone https://github.com/BerkIGuler/Assessment-Framework.git
2. Refer to run.py for examples of use of statistics class.


# Re-generating results of DFDC
## Selim Seferbekov
 
1. Clone https://github.com/selimsef/dfdc_deepfake_challenge repo in a local directory.
2. Build docker image using the DockerFile in that repo.

`   docker build -t <img_name> .   `

3. Run the Docker image by mounting pre-trained weight files and test data.

`   docker run --runtime=nvidia --ipc=host --rm --volume <current_directory>:/code --volume <test_videos_directory>:/test -it <docker_image> /bin/bash   `

4. Run predict_folder.py with correct path to weights.
```
   python predict_folder.py \
 --test-dir "/test" \
 --models final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36 \
  final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19 \
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29 \
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31 \
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37 \
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40 \
  final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23   
```

5. Transfer submission.csv file out of the container with docker cp command.

`   docker cp <container_name>:/code/submission.csv submission_sef.csv   `

## NtechLab

1. Clone https://github.com/NTech-Lab/deepfake-detection-challenge.git repo in a local directory.
2. Build docker image using the DockerFile in that repo.

`   docker build -t <img_name> .   `

3. Run the Docker image by mounting pre-trained weight files and test data.


`   docker run -d --runtime=nvidia --ipc=host --rm -d --volume <current_directory>:/code --volume <test_videos_directory>:/test -it <img_name> /bin/bash

4. Edit config.yaml file as below.
```
    DFDC_DATA_PATH: "test"
    ARTIFACTS_PATH: "/code/artifacts"
    MODELS_PATH: "/code/models"
    SUBMISSION_PATH: "/code/submission/submission.csv"
```
5. Run predict.py 


6. Transfer submission.csv file out of the container with docker cp command.

`   docker cp <container_name>:/code/submission/submission.csv submission_ntech.csv   `

## TeamVM

1. Clone https://github.com/cuihaoleo/kaggle-dfdc.git repo in a local directory.
2. load packages given in requirements.txt file
3. Edit paths to weights and test videos in submission.py
4. Run submission.py 

```
python3 submission.py
```


# Generating a Dataset to train DeepFake Detectors
Following script generates aligned and cropped faces from the videos in a source folder and saves these images in the target folder.

1. Clone https://github.com/cuihaoleo/kaggle-dfdc.git repo in a local directory.
2. load packages given in requirements.txt file
3. Run below command

```
python3 make_dataset.py <source_vids_folder> <output_folder>/
```


# Preprocessing of Test Videos

Refer to scripts in utils folder to encode videos in h.265 and to apply various filters.


