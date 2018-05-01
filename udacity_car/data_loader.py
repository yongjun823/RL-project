# pytorch 를 이용하여  데이터 학습시키자
# 데이터 로더, csv 내용 가져오기,
# 이미지 3장 모두 합셔서 배치로 만들자  + 스티어링 값

# csv 형식
# center img, right img, left img, 스티어링1, 스티어링2, 백, 가속력

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import csv
from PIL import Image

csv_path = 'D:\\tmp\\beta_simulator_windows\\R\\driving_log.csv'

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

pre_process = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def img_road(img_path):
    im = Image.open(img_path)
    img = pre_process(im)

    return img


class DriveDataLoader(data.Dataset):
    def __init__(self):
        f = open(csv_path, 'r', encoding='utf-8')
        csv_reader = csv.reader(f)

        self.drive_data = []

        for row in csv_reader:
            self.drive_data.append({
                'img_arr': [row[0], row[1], row[2]],
                'car_arr': [float(x) for x in [row[3], row[4], row[5]]]
            })

        f.close()

    def __getitem__(self, index):
        target_data = self.drive_data[index]

        center_img = img_road(target_data['img_arr'][0])
        left_img = img_road(target_data['img_arr'][1])
        right_img = img_road(target_data['img_arr'][2])

        tensors = torch.cat((center_img, left_img, right_img), 0)
        car_tensors = torch.from_numpy(np.array(target_data['car_arr']))
        return tensors, car_tensors

    def __len__(self):
        return len(self.drive_data)


custom_dataset = DriveDataLoader()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=100,
                                           shuffle=True)

for image, label in train_loader:
    print(label.shape)
    print(image.shape)
