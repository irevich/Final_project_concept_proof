import csv
import os
import numpy as np
import tensorflow as tf
import pywav
import soundfile
import glob

patient_diagnosis_dict = {}
diagnosis_list = []


##Audio files (.wav)

#Receives a list with all the .wav file, and set the bits per sample to 16 in case it is not
def transform_audio_filenames(audio_filenames):
    for audio_filename in audio_filenames:
      ##Check if its necessary to change the bits per sample to 16
      wav_file = pywav.WavRead(audio_filename)
      if(wav_file.getparams()['bitspersample']!=16):
        data, samplerate = soundfile.read(audio_filename)
        soundfile.write(audio_filename, data, samplerate, subtype='PCM_16')
      ##Check if we have to rename the files with the diagnosis
      posible_diagnosis = audio_filename.split('\\')[-1].split('_')[-1].split('.')[0]
      if(posible_diagnosis not in diagnosis_list):
        __rename_file_with_diagnosis(audio_filename)


#Receives a tensor of .wav files, and makes the preprocessing to get a tensorflow dataset of the form (spectogram, diagnosis index)
def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=tf.data.AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=tf.data.AUTOTUNE)
  return output_ds

#Receives the .wav filepath and returns the tuple (waveform, label)
def get_waveform_and_label(audio_filepath):
  label = __get_label(audio_filepath)
  audio_binary = tf.io.read_file(audio_filepath)
  waveform = __decode_audio(audio_binary)
  return waveform, label

#Receives an audio of a .wav file and his label (diagnosis), and returns the corresponding spectogram and index of the diagnosis in the diagnosis list
def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == diagnosis_list)
  return spectrogram, label_id

#Receives a .wav file on his waveform form, and returns his corresponding spectogram
def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

#Gets the diagnosis list
def get_diagnosis_list():
   return diagnosis_list

#Receives a diagnosis and the audio files path, and returns a random .wav filepath of that diagnosis, or None is the diagnosis provided is not valid
def get_random_file_of_diagnosis(diagnosis,audio_files_path):
   if(diagnosis not in diagnosis_list):
      return None
   dir_path = audio_files_path + os.path.sep + '*_' + diagnosis + '.wav'
   return np.random.choice(glob.glob(dir_path))

#Receive the filename of a .wav file, and rename it with his corresponding diagnosis
def __rename_file_with_diagnosis(audio_filename):
  audio_filename_parts = audio_filename.split(os.path.sep)
  audio_filename_without_directories = audio_filename_parts[-1]
  patient_id = int(audio_filename_without_directories.split('_')[0])
  diagnosis = patient_diagnosis_dict[patient_id]
  original_audio_filename_parts = audio_filename_without_directories.split('.')
  original_audio_filename = original_audio_filename_parts[0]
  original_extension = original_audio_filename_parts[1]
  directories = audio_filename_parts[:-1]
  new_audio_filename = os.path.sep.join(directories) + os.path.sep + original_audio_filename + '_' + diagnosis + '.' + original_extension
  os.rename(audio_filename,new_audio_filename)

#Receives the .wav filepath and returns the corresponding label for it as a tensor
def __get_label(audio_filepath):
    ##Separates directories
    file_first_parts = tf.strings.split(
      input=audio_filepath,
      sep=os.path.sep)
    ##Separates parts of the filename
    file_second_parts = tf.strings.split(
      input=file_first_parts[-1],
      sep='_')
    ##Gets the diagnosis
    file_third_parts = tf.strings.split(
      input=file_second_parts[-1],
      sep='.')
    return file_third_parts[0]



#Receives the binary version of the .wav file returns the audio a float tensor
def __decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

##CSVs

#Reads the diagnosis CSV file, and add items to the patient-diagnosis dictionary and diagnosis list above
def generate_diagnosis_data(diagnosis_csv_filepath,header=True):
    with open(diagnosis_csv_filepath, 'r', encoding= 'utf-8') as file:
        reader = csv.reader(file)
        if(header):
          next(reader) 
        for row in reader:
          patient_diagnosis_dict[int(row[0])] = row[1]
          if row[1] not in diagnosis_list:
              diagnosis_list.append(row[1])


