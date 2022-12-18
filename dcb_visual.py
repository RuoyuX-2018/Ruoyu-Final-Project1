import torch

from PIL import Image
from torchvision import transforms

dcb = torch.load('data/DCBs/HR/car/000000006810.pth.tar')
for i in range(dcb.shape[0]):
    print(dcb[i].max())
    t1 = transforms.ToPILImage()
    t2 = transforms.Resize((1028, 1028))
    img = t1(dcb[i])
    img = t2(img)
    img.save('result/test_dcb/'+str(i)+'.jpg')
