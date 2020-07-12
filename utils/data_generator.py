import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging

import config
from utilities import int16_to_float32


class GtzanDataset(object):
    def __init__(self):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        Args:
          clip_samples: int
          classes_num: int
        """
        pass
    
    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}
        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            waveform = int16_to_float32(hf['waveform'][index_in_hdf5])
            target = hf['target'][index_in_hdf5].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict


class Base(object):
    def __init__(self, indexes_hdf5_path, batch_size, random_seed):
        """Base class of train sampler.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Load target
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            # self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
            self.folds = hf['fold'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.targets.shape
        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class TrainSampler(object):
    def __init__(self, hdf5_path, holdout_fold, batch_size, random_seed=1234):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        # super(TrainSampler, self).__init__(indexes_hdf5_path, batch_size, 
            # random_seed)

        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        with h5py.File(hdf5_path, 'r') as hf:
            self.folds = hf['fold'][:].astype(np.float32)

        self.indexes = np.where(self.folds != int(holdout_fold))[0]
        self.audios_num = len(self.indexes)
        # self.validate_audio_indexes = np.where(self.folds == int(holdout_fold))[0]
        
        # self.indexes = np.arange(self.audios_num)
            
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.indexes[self.pointer]})
                i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state
            
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class EvaluateSampler(object):
    def __init__(self, hdf5_path, holdout_fold, batch_size, random_seed=1234):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        # super(TrainSampler, self).__init__(indexes_hdf5_path, batch_size, 
            # random_seed)

        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

        with h5py.File(hdf5_path, 'r') as hf:
            self.folds = hf['fold'][:].astype(np.float32)

        self.indexes = np.where(self.folds == int(holdout_fold))[0]
        self.audios_num = len(self.indexes)
        
    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for i in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.indexes[i]})

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict


# class Base(object):
    
#     def __init__(self):
#         '''Base class for data generator
#         '''
#         pass

#     def load_hdf5(self, hdf5_path):
#         '''Load hdf5 file. 
        
#         Returns:
#           data_dict: dict of data, e.g.:
#             {'audio_name': np.array(['a.wav', 'b.wav', ...]), 
#              'feature': (audios_num, frames_num, mel_bins)
#              'target': (audios_num,), 
#              ...}
#         '''
#         data_dict = {}
        
#         with h5py.File(hdf5_path, 'r') as hf:
#             data_dict['audio_name'] = np.array(
#                 [audio_name.decode() for audio_name in hf['audio_name'][:]])

#             data_dict['waveform'] = hf['waveform'][:]
#             data_dict['target'] = hf['target'][:].astype(np.float32)
#             data_dict['fold'] = hf['fold'][:].astype(np.int32)
            
#         return data_dict

#     def transform(self, x):
#         return scale(x, self.scalar['mean'], self.scalar['std'])


# class DataGenerator(Base):
    
#     def __init__(self, hdf5_path, holdout_fold, batch_size):
#         '''Data generator for training and validation. 
        
#         Args:
#           feature_hdf5_path: string, path of hdf5 feature file
#           train_csv: string, path of train csv file
#           validate_csv: string, path of validate csv file
#           holdout_fold: set 1 for development and none for training 
#               on all data without validation
#           scalar: object, containing mean and std value
#           batch_size: int
#           seed: int, random seed
#         '''
 
#         self.batch_size = batch_size
#         self.random_state = np.random.RandomState(random_seed)
        
#         # self.classes_num = classes_num
#         self.all_classes_num = len(config.labels)
#         self.lb_to_idx = config.lb_to_idx
#         self.idx_to_lb = config.idx_to_lb
        
#         # Load training data
#         load_time = time.time()
        
#         self.data_dict = self.load_hdf5(hdf5_path)

#         self.train_audio_indexes = np.where(self.data_dict['fold'] != int(holdout_fold))[0]
#         self.validate_audio_indexes = np.where(self.data_dict['fold'] == int(holdout_fold))[0]

#         if few_shots > 0:
#             self.random_state.shuffle(self.train_audio_indexes)
#             classes_num = self.data_dict['weak_target'].shape[-1]
#             new_indexes = []
#             for k in range(classes_num):
#                 new_indexes.append(self.train_audio_indexes[np.where(
#                     self.data_dict['weak_target'][self.train_audio_indexes][:, k] == 1)[0][0 : few_shots]])
#             self.train_audio_indexes = np.concatenate(new_indexes)

#         logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
#         logging.info('Training audio num: {}'.format(len(self.train_audio_indexes)))            
#         logging.info('Validation audio num: {}'.format(len(self.validate_audio_indexes)))
        
#         self.random_state.shuffle(self.train_audio_indexes)
#         self.pointer = 0
        
#     def generate_train(self):
#         '''Generate mini-batch data for training. 
        
#         Returns:
#           batch_data_dict: dict containing audio_name, feature and target
#         '''

#         while True:
#             # Reset pointer
#             if self.pointer >= len(self.train_audio_indexes):
#                 self.pointer = 0
#                 self.random_state.shuffle(self.train_audio_indexes)

#             # Get batch audio_indexes
#             batch_audio_indexes = self.train_audio_indexes[
#                 self.pointer: self.pointer + self.batch_size]
                
#             self.pointer += self.batch_size

#             batch_data_dict = {}

#             batch_data_dict['audio_name'] = \
#                 self.data_dict['audio_name'][batch_audio_indexes]
            
#             batch_data_dict['waveform'] = int16_to_float32(self.data_dict['waveform'][batch_audio_indexes])
#             batch_data_dict['weak_target'] = self.data_dict['weak_target'][batch_audio_indexes]
            
#             yield batch_data_dict

#     def generate_validate(self, data_type, source, max_iteration=None):
#         '''Generate mini-batch data for training. 
        
#         Args:
#           data_type: 'train' | 'validate'
#           source: 'a' | 'b' | 'c'
#           max_iteration: int, maximum iteration to validate to speed up validation
        
#         Returns:
#           batch_data_dict: dict containing audio_name, feature and target
#         '''
        
#         batch_size = self.batch_size
        
#         if data_type == 'train':
#             audio_indexes = np.array(self.train_audio_indexes)
#         elif data_type == 'validate':
#             audio_indexes = np.array(self.validate_audio_indexes)
#         else:
#             raise Exception('Incorrect argument!')
            
#         iteration = 0
#         pointer = 0
        
#         while True:
#             if iteration == max_iteration:
#                 break

#             # Reset pointer
#             if pointer >= len(audio_indexes):
#                 break

#             # Get batch audio_indexes
#             batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]                
#             pointer += batch_size
#             iteration += 1

#             batch_data_dict = {}

#             batch_data_dict['audio_name'] = \
#                 self.data_dict['audio_name'][batch_audio_indexes]
            
#             batch_data_dict['waveform'] = int16_to_float32(self.data_dict['waveform'][batch_audio_indexes])
#             batch_data_dict['weak_target'] = self.data_dict['weak_target'][batch_audio_indexes]
            
#             yield batch_data_dict
