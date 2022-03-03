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

class Rescale(object):
    """
        Rescale the image in a sample to a given size
    
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same
    """
    def __init__(self,output_size):
        assert(isinstance(output_size, (int, tuple)))
        self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'],sample['landmarks']

        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))   #缩放图像
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]
        print("     Rescale")
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """ Crop randomly the image in a sample
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self,output_size):
       assert isinstance(output_size, (int, tuple))
       if(isinstance(output_size, int)):
            self.output_size = (output_size, output_size)
       else:
            assert(len(output_size) == 2)
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]
        print("     RandomCrop")
        return {'image': image, 'landmarks': landmarks}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        print("     ToTensor")
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

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
'''
'''
    下面示例如何缩放和裁剪和转换成tensor图像,每一次跌倒都执行了Rescale,RandomCrop,ToTensor
'''
transformed_dataset = FaceLandmarksDataset(csv_file='E:/blowfish_github/data/faces/face_landmarks.csv',
                                           root_dir='E:/blowfish_github/data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
'''
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())
'''
'''
    # However, we are losing a lot of features by using a simple for loop to iterate over the data. 
    # In particular, we are missing out on:

    # 1.  Batching the data
    # 2.  Shuffling the data
    # 3.  Load the data in parallel using multiprocessing workers.

    # torch.utils.data.DataLoader is an iterator which provides all these features. 
    # Parameters used below should be clear. One parameter of interest is collate_fn. 
    # You can specify how exactly the samples need to be batched using collate_fn. However, 
    # default collate should work fine for most use cases.

    # FaceLandmarksDataset_transform_Iterating_batch.py是按批次迭代的代码
'''
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

#Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

# if you are using Windows, uncomment the next line and indent the for loop.
# you might need to go back and change "num_workers" to 0.

# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
'''
    问题
    1.数据很大怎么办，现在是一次读取全部
    2.如何遍历全部呢？
    如下链接不知道能解决不
    https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
'''
    