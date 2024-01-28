import numpy as np
import librosa
from torch.utils.data import Dataset
from .dataaug import spec_augment
from .utils2 import extract_melspectrogram, downsample_spectrogram

SAMPLING_RATE = 8000

class ourdata(Dataset):
    def __init__(self, files):
        self.files = files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        x, sr = librosa.load(filename, sr = SAMPLING_RATE)
        melspectrogram = extract_melspectrogram(x, sr, num_mels=13)
        down_spectrogram = downsample_spectrogram(melspectrogram, 25)
        
        inp = [spec_augment(np.expand_dims(down_spectrogram,0)).squeeze().transpose(1,0), 
               spec_augment(np.expand_dims(down_spectrogram,0)).squeeze().transpose(1,0)]
        return inp, 1



class ContrastiveLearningDataset:
    def __init__(self):
        pass

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        return

    def get_dataset(self, files):
        train_set = ourdata(files)
        return train_set