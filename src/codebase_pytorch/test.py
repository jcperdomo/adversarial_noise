from models.vgg import vgg16, vgg19
from utils.adversarialDataSet import AdversarialImageSet
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.helper import UnNormalize
import matplotlib.pyplot as plt
import numpy as np


def show(img, title):
    npimg = img.numpy()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

unnormalize = UnNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Scale((224, 224)),  # Inputs need to rescaled to 224 * 224
   transforms.ToTensor(),
   normalize
])


data = AdversarialImageSet("../test_data/", "../image_label_target.csv", transform=preprocess)


loader = DataLoader(data, batch_size=4, shuffle=False, num_workers=4)
iterator = iter(loader)
images, true_labels, target_labels = next(iterator)

model = vgg16(pretrained=True)
s = model(Variable(images))
print s.max(1)
grads = model.get_gradient(images, true_labels)
epsilon = .1


save_image(images, "normal.jpeg")

noised = images + epsilon * grads.sign()
save_image(noised, "noised.jpeg")


