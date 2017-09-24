import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

default_transform = transforms.Compose([
   transforms.Scale((256, 256)),
   transforms.ToTensor()
])

class AdversarialImageSet(Dataset):

    def __init__(self, img_dir, labelCSVfile, transform=default_transform):
        self.img_dir = img_dir
        self.imgInfo = pd.read_csv(labelCSVfile)
        self.transform = transform

    def __getitem__(self, index):
        """
        returns:

        img = pytorch tensor of image with transforms applied
        true label = int corresponding to ground truth
        target label = int corresponding to target class
        """
        name, true_label, target_label = self.imgInfo.iloc[index][['image name', 'true label id', 'target label id']]
        img = Image.open(self.img_dir + name).convert('RGB')
        return self.transform(img), true_label, target_label

    def __len__(self):
        return self.imgInfo.shape[0]

    def get_info(self, index):
        """
        returns Pandas DataFrame row with information about the test sample
        """
        return self.imgInfo.iloc[index]

