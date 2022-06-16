# Assessment-Framework
Set of scripts written to calculate a few statistics to evaluate the performance of classifiers. 


# Re-generating results for DFDC
## Selim Seferbekov
 
1. Clone https://github.com/selimsef/dfdc_deepfake_challenge repo in a local directory.
2. Build docker image using the DockerFile in that repo.
3. Run the Docker image by mounting pre-trained weight files and test data.
4. Run predict_folder.py with correct path to weights.
5. Transfer submission.csv file out of the container with docker cp command.

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


# Running the Assessment Framework for DFDC

