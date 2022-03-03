from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

'''
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self,csv_file,root_dir,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):          #returns the size of the dataset.
        return len(self.landmarks_frame)

    def __getitem__(self, idx): #support the indexing such that dataset[i] can be used to get iiith sample.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx,0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx,1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}   #sample is dict type
        if self.transform:
            sample = self.transform(sample)

        return sample

#使用这个类
face_dataset = FaceLandmarksDataset(csv_file='E:/blowfish_github/data/faces/face_landmarks.csv',
                                    root_dir='E:/blowfish_github/data/faces/')
fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i,sample['image'].shape,sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break

#Transforms
'''
    我们注意到一个问题，上面的samples不是同样的大小，大多数神经网络期望图像有相同的大小.
    因此，我们需要写预处理代码，我们创建三个Transforms
    1.  Rescale : to scale the image 裁剪图像
    2.  RandomCrop :to crop from image randomly,This is data augmentation
    3.  ToTensor:   to convert the numpy image to torch images(we need to swap axes)
    
    We will write them as callable classes instead of simple functions so that parameters of the transform need not be passed everytime it’s called. For this, we just need to implement __call__ method and if required, __init__ method. We can then use a transform like this:
        tsfm = Transform(params)
    transformed_sample = tsfm(sample)

    FaceLandmarksDataset_transform.py实现上面的说明
'''