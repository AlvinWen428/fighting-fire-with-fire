import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate

import numpy as np
from PIL import Image
import cv2
from copy import deepcopy
from tqdm import tqdm

from BASNet.data_loader import RescaleT
from BASNet.data_loader import ToTensorLab

from BASNet.model import BASNet

from BASNet.basnet_test import normPRED


class NicoDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        self.image_path_list = []
        data_partition_list = os.listdir(self.image_folder)
        for data_partition in data_partition_list:  # train, val, test
            image_names = os.listdir(os.path.join(self.image_folder, data_partition))
            self.image_path_list.extend([os.path.join(self.image_folder, data_partition, name) for name in image_names])

    def __getitem__(self, index):
        file_name = self.image_path_list[index]
        image = Image.open(file_name).convert('RGB')
        image = torch.Tensor(np.array(image))

        label_3 = np.zeros(image.shape)

        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['img_path'] = file_name

        return sample

    def __len__(self):
        return len(self.image_path_list)

    @staticmethod
    def collate(batch):
        img_paths = [sample.pop('img_path') for sample in batch]
        collated_batch = default_collate(batch)
        collated_batch['img_path'] = img_paths
        return collated_batch


def crop(image, saliency):
    """
    image: np.array [w, h, 3]
    saliency: np.array [w, h]
    """
    # find the smallest bbox
    leftmost, rightmost, upmost, lowmost = 0, 0, 0, 0
    for i in range(saliency.shape[0]):
        if np.any(saliency[i, ...] > 0):
            upmost = i
            break
    for i in range(saliency.shape[0]-1, -1, -1):
        if np.any(saliency[i, ...] > 0):
            lowmost = i
            break
    for i in range(saliency.shape[1]):
        if np.any(saliency[:, i, ...] > 0):
            leftmost = i
            break
    for i in range(saliency.shape[1]-1, -1, -1):
        if np.any(saliency[:, i, ...] > 0):
            rightmost = i
            break

    output = image[upmost: lowmost, leftmost: rightmost]
    return output


def postprocess(model_output: np.array) -> np.array:
    """
	We postprocess the predicted saliency mask to remove very small segments.
	If the mask is too small overall, we skip the image.

	Args:
	    model_output: The predicted saliency mask scaled between 0 and 1.
	                  Shape is (height, width).
	Return:
            result: The postprocessed saliency mask.
    """
    mask = (model_output > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(deepcopy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Throw out small segments
    for contour in contours:
        segment_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        segment_mask = cv2.drawContours(segment_mask, [contour], 0, 255, thickness=cv2.FILLED)
        area = (np.sum(segment_mask) / 255.0) / np.prod(segment_mask.shape)
        if area < 0.01:
            mask[segment_mask == 255] = 0

    # If area of mask is too small, return None
    if np.sum(mask) / np.prod(mask.shape) < 0.01:
        return model_output

    return mask


def main(image_dir, output_folder):
    model_dir = './BASNet/saved_models/upsupervised_basnet/basnet_unsupervised_option.pth.tar'

    transform = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])
    test_dataset = NicoDataset(image_dir, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4,
                                 collate_fn=test_dataset.collate)

    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_dir)['net'])
    net.cuda()
    net.eval()

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            img_path_list = batch['img_path']
            input_images = batch['image']

            input_images = input_images.type(torch.FloatTensor)
            input_images = input_images.cuda()

            d1, d2, d3, d4, d5, d6, d7, d8 = net(input_images)

            # normalization
            pred = d1[:, 0, :, :]

            for i, pred_i in enumerate(pred):

                pred_i = normPRED(pred_i)
                saliency_mask = postprocess(pred_i.detach().cpu().numpy())

                if img_path_list[i].split('/')[-1].split('.')[0] == '1_15_22':
                    saliency_mask = np.zeros_like(saliency_mask)
                    saliency_mask[37:225, 37:225] = 1

                saliency_mask = saliency_mask*255

                saliency_mask = Image.fromarray(np.uint8(saliency_mask))

                # resize
                raw_image = Image.open(img_path_list[i]).convert('RGB')
                saliency_mask = saliency_mask.resize(raw_image.size, resample=Image.BILINEAR)

                # binary
                saliency_mask = np.array(saliency_mask)
                saliency_mask = (saliency_mask > 0).astype(np.uint8)

                output_image = crop(np.array(raw_image), saliency_mask)

                partition_name = img_path_list[i].split('/')[-2]
                img_name = img_path_list[i].split('/')[-1].split('.')[0]
                img_name = img_name + '.png'
                output_image_file = Image.fromarray(np.uint8(output_image))
                os.makedirs(os.path.join(output_folder, partition_name), exist_ok=True)
                output_image_file.save(os.path.join(output_folder, partition_name, img_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='/data/cwen/NICO/multi_classification/')
    parser.add_argument('--output-folder', type=str, default='/data/cwen/NICO/key_inputs')
    args = parser.parse_args()

    main(args.image_dir, args.output_folder)
