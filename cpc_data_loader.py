# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The data loader for the audio CPC model.

"""

import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
import os
import sys
import random
import librosa
import torchaudio


     

class CPC_raw_audio_dataset(Dataset):
    
    """
    A raw audio dataset (a torch.utils.data.Dataset object) for a CPC model, practically the same
    as in the original CPC paper.
    
    _____________________________________________________________________________________________
    Input parameters:
        
        
    train_val_test: A string to determine whether we are creating a train, validation, or test
                    set. Options: 'train', 'validation', or 'test'
            
    file_dir: The directory of the WAV files.
    
    audio_window_length: The number of samples in each audio window.
             
    Fs: The sampling rate of our WAV files.
    
    train_test_ratio: The ratio in which the files in file_dir are split into a training and test
                      set. E.g., a ratio of 0.8 will assign 80% of the data for training and 20%
                      for testing.
                      
    train_val_ratio: The ratio in which the training files are split into a training and
                     validation set. E.g., a ratio of 0.75 will split the training data so that
                     75% of the data is used for training and 25% is used for validation.
                     
    random_seed: A random seed for making our random data splits consistent between experiments.
    _____________________________________________________________________________________________
    
    """

    def __init__(self, train_val_test='train', file_dir='./wav_files', audio_window_length=20480, Fs=16000,
                 train_test_ratio=0.8, train_val_ratio=0.75, random_seed=22):
        super().__init__()
        
        self.audio_window_length = audio_window_length
        
        # Find out our WAV files in the given directory
        try:
            filenames_wav = os.listdir(file_dir)
        except FileNotFoundError:
            sys.exit('Given .wav file directory ' + file_dir + ' does not exist!')
        
        # Clean the list if there are other files than .wav files
        wav_file_names = [filename for filename in filenames_wav if filename.endswith('.wav')]
        del filenames_wav
        
        # We only take into account the audio files that are longer than audio_window_length
        wav_files = []
        for file in wav_file_names:
            x, fs = librosa.core.load(os.path.join(file_dir, file), sr=Fs)
            if len(x) > audio_window_length:
                wav_files.append(x)
        
        wav_files = np.array(wav_files, dtype=object)
        
        # Split our data into a train, validation, and test set
        np.random.seed(random_seed)
        mask_traintest_split = np.random.rand(len(wav_files)) <= train_test_ratio
        trainval_files = wav_files[mask_traintest_split]
        test_files = wav_files[~mask_traintest_split]
        np.random.seed(2*random_seed)    # We use a different random seed for splitting trainval_files
        mask_trainval_split = np.random.rand(len(trainval_files)) <= train_val_ratio
        train_files = trainval_files[mask_trainval_split]
        val_files = trainval_files[~mask_trainval_split]
        
        del wav_files
        del trainval_files
        
        # train_val_test has three options: 'train', 'validation' and 'test'
        if train_val_test == 'train':
            self.feats = train_files
        elif train_val_test == 'validation':
            self.feats = val_files
        else:
            self.feats = test_files
        

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, index):
        # we take a random part of the audio file of length audio_window_length
        part_index = np.random.randint(len(self.feats[index]) - self.audio_window_length + 1)
        
        return from_numpy(self.feats[index][part_index:(part_index + self.audio_window_length)]), []


class CPC_raw_audio_dataset_with_labels(Dataset):
    
    """
    A raw audio dataset (a torch.utils.data.Dataset object) for a CPC model, practically the same
    as in the original CPC paper.
    
    _____________________________________________________________________________________________
    Input parameters:
        
        
    train_val_test: A string to determine whether we are creating a train, validation, or test
                    set. Options: 'train', 'validation', or 'test'
            
    file_dir: The directory of the WAV files.
    
    audio_window_length: The number of samples in each audio window.
             
    Fs: The sampling rate of our WAV files.
    
    train_test_ratio: The ratio in which the files in file_dir are split into a training and test
                      set. E.g., a ratio of 0.8 will assign 80% of the data for training and 20%
                      for testing.
                      
    train_val_ratio: The ratio in which the training files are split into a training and
                     validation set. E.g., a ratio of 0.75 will split the training data so that
                     75% of the data is used for training and 25% is used for validation.
                     
    random_seed: A random seed for making our random data splits consistent between experiments.
    _____________________________________________________________________________________________
    
    """

    def __init__(self, train_val_test='train', file_dir='./wav_files', audio_window_length=20480, Fs=16000,
                 train_test_ratio=0.9, train_val_ratio=0.9, random_seed=22):
        super().__init__()
        
        self.audio_window_length = audio_window_length
        
        # Find out our WAV files in the given directory
        try:
            filenames_wav = os.listdir(file_dir)
        except FileNotFoundError:
            sys.exit('Given .wav file directory ' + file_dir + ' does not exist!')
        
        # Clean the list if there are other files than .wav files
        wav_file_names = [filename for filename in filenames_wav if filename.endswith('.wav')]
        del filenames_wav
        
        # We only take into account the audio files that are longer than audio_window_length
        wav_files = []
        speaker_ids = []
        for file in wav_file_names:
            speaker_id, chapter_id, utterance = file.split('-')
            speaker_ids.append(speaker_id)
            x, fs = librosa.core.load(os.path.join(file_dir, file), sr=Fs)
            if len(x) > audio_window_length:
                wav_files.append(x)

        unique_speakers, labels = np.unique(speaker_ids, return_inverse=True)
        wav_files = np.array(wav_files, dtype=object)
        
        # Split our data into a train, validation, and test set
        np.random.seed(random_seed)
        mask_traintest_split = np.random.rand(len(wav_files)) <= train_test_ratio
        trainval_files = wav_files[mask_traintest_split]
        trainval_labels = labels[mask_traintest_split]
        test_files = wav_files[~mask_traintest_split]
        test_labels = labels[~mask_traintest_split]
        np.random.seed(2*random_seed)    # We use a different random seed for splitting trainval_files
        mask_trainval_split = np.random.rand(len(trainval_files)) <= train_val_ratio
        train_files = trainval_files[mask_trainval_split]
        train_labels = trainval_labels[mask_trainval_split]
        val_files = trainval_files[~mask_trainval_split]
        val_labels = trainval_labels[~mask_trainval_split]
        
        del wav_files
        del trainval_files
        del trainval_labels
        
        # train_val_test has three options: 'train', 'validation' and 'test'
        if train_val_test == 'train':
            self.feats = train_files
            self.labels = train_labels
        elif train_val_test == 'validation':
            self.feats = val_files
            self.labels = val_labels
        else:
            self.feats = test_files
            self.labels = test_labels
        

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, index):
        # we take a random part of the audio file of length audio_window_length
        part_index = np.random.randint(len(self.feats[index]) - self.audio_window_length + 1)
        
        return from_numpy(self.feats[index][part_index:(part_index + self.audio_window_length)]), self.labels[index]


class CPCDataset(Dataset):
    """
    A raw audio dataset for a CPC model, loading files from the LibriSpeech folder and performing splits
    between train, validation, and test based on filenames.
    _____________________________________________________________________________________________
    """

    def __init__(self, train_val_test='train', file_dir='./LibriSpeech/train-clean-100', 
                 audio_window_length=20480, Fs=16000, train_test_ratio=0.9, train_val_ratio=0.9, 
                 random_seed=42, num_speakers=None):
        super().__init__()
        
        self.audio_window_length = audio_window_length
        self.file_dir = file_dir
        self.Fs = Fs
        self.num_speakers = num_speakers
        
        # Initialize random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Traverse the directory structure and collect file paths
        self.file_paths = []
        self.speaker_ids = []
        
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.endswith('.flac'):  # Assuming LibriSpeech files are .flac
                    speaker_id, chapter_id, utterance_id = file.split('-')[:3]
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)
                    self.speaker_ids.append(speaker_id)

        # Convert speaker IDs to labels
        unique_speakers, labels = np.unique(self.speaker_ids, return_inverse=True)

        # Filter to select only `num_speakers` if provided
        if self.num_speakers is not None:
            if self.num_speakers > len(unique_speakers):
                raise ValueError(f"Requested {self.num_speakers} speakers, but only {len(unique_speakers)} are available.")
            
            # Randomly select `num_speakers` speakers
            selected_speakers = np.random.choice(unique_speakers, self.num_speakers, replace=False)
            
            # Filter file paths and labels based on selected speakers
            selected_indices = [i for i, speaker_id in enumerate(self.speaker_ids) if speaker_id in selected_speakers]
            self.file_paths = [self.file_paths[i] for i in selected_indices]
            self.labels = [labels[i] for i in selected_indices]
        else:
            self.labels = labels  # Use all speakers if no limit is set
        
        # Split file paths into train, validation, and test sets
        num_files = len(self.file_paths)
        indices = np.arange(num_files)
        np.random.shuffle(indices)

        train_test_split = int(train_test_ratio * num_files)
        train_val_split = int(train_val_ratio * train_test_split)
        
        if train_val_test == 'train':
            self.file_paths = [self.file_paths[i] for i in indices[:train_val_split]]
            self.labels = [self.labels[i] for i in indices[:train_val_split]]
        elif train_val_test == 'validation':
            self.file_paths = [self.file_paths[i] for i in indices[train_val_split:train_test_split]]
            self.labels = [self.labels[i] for i in indices[train_val_split:train_test_split]]
        else:
            self.file_paths = [self.file_paths[i] for i in indices[train_test_split:]]
            self.labels = [self.labels[i] for i in indices[train_test_split:]]
        
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load the audio file at the given index
        file_path = self.file_paths[index]
        waveform, sample_rate = torchaudio.load(file_path)
        # Resample the audio if necessary
        if sample_rate != self.Fs:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.Fs)(waveform)

        # Select a random window from the audio file
        if waveform.size(1) > self.audio_window_length:
            start = np.random.randint(0, waveform.size(1) - self.audio_window_length)
            waveform = waveform[:, start:start + self.audio_window_length]
        
        waveform = waveform.squeeze(0)

        return waveform, self.labels[index]



class CPC_logmel_dataset(Dataset):
    
    """
    A dataset (torch.utils.data.Dataset object) for a CPC model that first converts WAV files
    into log-mel features.
    
    _____________________________________________________________________________________________
    Input parameters:
        
        
    train_val_test: A string to determine whether we are creating a train, validation, or test
                    set. Options: 'train', 'validation', or 'test'
            
    file_dir: The directory of the WAV files.
             
    Fs: The sampling rate of our WAV files.
    
    num_logmel_frames: The number of log-mel frames that we use as features (we want to have
                             constant-length features)
    
    train_test_ratio: The ratio in which the files in file_dir are split into a training and test
                      set. E.g., a ratio of 0.8 will assign 80% of the data for training and 20%
                      for testing.
                      
    train_val_ratio: The ratio in which the training files are split into a training and
                     validation set. E.g., a ratio of 0.75 will split the training data so that
                     75% of the data is used for training and 25% is used for validation.
                     
    random_seed: A random seed for making our random data splits consistent between experiments.
    _____________________________________________________________________________________________
    
    """

    def __init__(self, train_val_test='train', file_dir='./wav_files', Fs=16000, num_logmel_frames=128,
                 train_test_ratio=0.8, train_val_ratio=0.75, random_seed=22):
        super().__init__()
        
        self.num_logmel_frames = num_logmel_frames
        
        # Find out our WAV files in the given directory
        try:
            filenames_wav = os.listdir(file_dir)
        except FileNotFoundError:
            sys.exit('Given .wav file directory ' + file_dir + ' does not exist!')
        
        # Clean the list if there are other files than .wav files
        wav_file_names = [filename for filename in filenames_wav if filename.endswith('.wav')]
        del filenames_wav
        
        # We only take into account the log-mels that are longer than num_logmel_frames
        logmels = []
        for file in wav_file_names:
            x, fs = librosa.core.load(os.path.join(file_dir, file), sr=Fs)
                
            # The number of FFT points for a window length of 30 ms and 10 ms shifts
            num_fft = int(0.03 * fs)
            shift = int(0.01 * fs)
            
            # Extract the log-mel spectrogram
            melspec = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=num_fft, hop_length=shift, 
                                                     n_mels=40)
            
            logmel = librosa.core.power_to_db(melspec)
            
            if logmel.shape[1] > self.num_logmel_frames:
                logmels.append(logmel.T)
                
        logmels = np.array(logmels, dtype=object)
        
        # Split our data into a train, validation, and test set
        np.random.seed(random_seed)
        mask_traintest_split = np.random.rand(len(logmels)) <= train_test_ratio
        trainval_files = logmels[mask_traintest_split]
        test_files = logmels[~mask_traintest_split]
        np.random.seed(2*random_seed)    # We use a different random seed for splitting trainval_files
        mask_trainval_split = np.random.rand(len(trainval_files)) <= train_val_ratio
        train_files = trainval_files[mask_trainval_split]
        val_files = trainval_files[~mask_trainval_split]
        
        del logmels
        del trainval_files
        
        # train_val_test has three options: 'train', 'validation' and 'test'
        if train_val_test == 'train':
            self.feats = train_files
        elif train_val_test == 'validation':
            self.feats = val_files
        else:
            self.feats = test_files


    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, index):
        # we take a random part of the log-mel features of length num_logmel_frames
        part_index = np.random.randint(len(self.feats[index]) - self.num_logmel_frames + 1)
        
        return from_numpy(self.feats[index][part_index:(part_index + self.num_logmel_frames)]), []
