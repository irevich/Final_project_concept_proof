# 72.45 Final Project

## Final project concept proof - WAV files preprocessing for training an IA model

### Instituto Tecnológico de Buenos Aires (ITBA)

## Autor

- [Igal Leonel Revich](https://github.com/irevich)

## Index

- [72.45 Final Project](#7245-final-project)
  - [Final project concept proof - WAV files preprocessing for training an IA model](#final-project-concept-proof---wav-files-preprocessing-for-training-an-ia-model)
    - [Instituto Tecnológico de Buenos Aires (ITBA)](#instituto-tecnológico-de-buenos-aires-itba)
  - [Autor](#autor)
  - [Index](#index)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [IA Model with audio files tutorial](#ia-model-with-audio-files-tutorial)
    - [Description](#description)
    - [Dataset](#dataset)
  - [Proof concept](#proof-concept)
    - [Description](#description-1)
    - [Constants](#constants)
    - [Dataset](#dataset-1)
    - [Final test](#final-test)
  

## Introduction

  This repository was created in order to make a concept proof for the final project of career that is supposed to be made in one year. The main idea was to start making a little preprocessing of the .wav files that the project will be using, train some existing IA model with those and build some metrics of the model learning. For doing that, 2 Jupyter notebooks have been created:
  
  - <i> training_ia_model_with_audio_files_tutorial.ipynb </i>
  - <i>concept_proof.ipynb</i>
   
  The first one is the tutorial that has been followed to learn about how to process .wav files for training some IA model for prediction, and the second one that is the corresponding concept proof was adapting the tutorial code to the dataset that is going to be used in the final project (audios of people breathing for detecting breathing deseases). Both of them are explained in more detail in the following sections.

## Dependencies

In order to execute correctly both notebooks, you must download the corresponding dependencies through the <i>requirements.txt</i> by executing this code:

```
pip install -r requirements.txt
```

After doing that, both notebooks are ready to be executed.

## IA Model with audio files tutorial

### Description

The first notebook that is mentioned above consist on a code where downloads a dataset of audio files with .wav format of different english keywords ("yes", "no", etc), and those audio files are preprocessed for giving them to an IA model for training it so later it can predict the corresponding keyword from a given word. The IA model chosen for this code is a CNN, and it works by converting the .wav files from waveforms to spectograms. The code was obtained by following this tutorial: https://www.tensorflow.org/tutorials/audio/simple_audio?hl=es-419.

### Dataset

The corresponding dataset is downloaded in the first code block of the notebook, and it downloades it the path specified in the variable DATASET_PATH of that block, inside the repo's folder (by default is data/mini_speech_commands).

## Proof concept

### Description

The proof concept is basically the notebook made from the tutorial applied to the dataset that is going to be used for the project, that are the breathing audios of people (also in .wav format), so then the IA model can be trained with those and their corresponding diagnosis for later detecting different breathing deseases by receiving another breathing audio.

### Constants

The first code block of the notebook consist of the constants that are going to be used in all the code. These are the following:

- <i>SEED</i>: Corresponding seed use for randomness.
- Paths
  - <i>DATASET_AUDIO_PATH</i>: The path inside the repo's folder where the corresponding .wav files will be. By default is "dataset/audio_and_txt_files". It can be changed but you must move the .wav files first.
  - <i>DATASET_LABELS_PATH </i>: The path inside the repo's folder where the csv with the diagnosis for each audio will be. By default is "dataset/patient_diagnosis.csv". It can be changed but you must move the csv first (and also if you change the name of the csv in the path you must change it in the file as well).
- IA Model
  - <i>TRAINING_PERCENTAGE </i>: Percentage of the dataset that will be used for training. It must sum 100 with <i>VALIDATION_PERCENTAGE </i> and <i>TESTING_PERCENTAGE </i>.
  - <i>VALIDATION_PERCENTAGE </i>: Percentage of the dataset that will be used for validation. It must sum 100 with <i>TRAINING_PERCENTAGE </i> and <i>TESTING_PERCENTAGE </i>.
  - <i>TESTING_PERCENTAGE </i>: Percentage of the dataset that will be used for testing. It must sum 100 with <i>VALIDATION_PERCENTAGE </i> and <i>TRAINING_PERCENTAGE </i>.
  - <i>EPOCHS </i>: The epochs quantity that the model will be trained. Must be a positive integer number.

### Dataset

For preprocessing the corresponding dataset and the give it to the IA model for training, it was necessary to make some changes to the .wav files inside:
- In case the bits per sample wasn't 16, it was changed to that value, because Tensorflow method por decode .wav files can only decode those that have 16 as bits per sample.
- In the filename of each .wav file, the diagnosis was appended (with the format "_diagnosis"), in order to get the corresponding label for training on a similar way it was made in the tutorial.

Because of that transformation (made by the function <i>transform_audio_filenames(audio_filenames)</i>), the first time the notebook is executed it can delay a while the corresponding block of code (almost 3 min), but then the following times this transformation is not necessary beacause files are already transformed, so next executions doesn't delay so much.

### Final test
In the last test made for the model, that is made in the last code block and it produces a bar plot, it chooses randomly an audio file of some diagnosis, and it plots the prediction of the corresponding model for that audio file. The used diagnosis is set in the variable <i>test_diagnosis</i> and it can be changed, but it must be one of the valid diagnosis, and written as it figures in the diagnosis list printed in the fourth block of code.



