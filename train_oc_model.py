import torch
import torchvision
import os
import matplotlib.pyplot as plt
import utils

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from tqdm import trange
from datetime import datetime
from ochumanApi.ochuman import Poly2Mask, OCHuman
from torch.utils.data import Dataset
from google.colab.patches import cv2
from torch import tensor
from torchvision import tv_tensors

plt.rcParams['figure.figsize'] = (15, 15)
torch.manual_seed(111111)

ochuman = OCHuman(AnnoFile='./ochuman.json', Filter='segm')


class OCHumanDataset(Dataset):
    def __init__(self, image_root: str, oc_human: OCHuman):
        self.root = image_root
        self.data = oc_human.loadImgs(imgIds=oc_human.getImgIds())
        self.images, self.masks, self.bounding_boxes = self.__get_properties(oc_human_data=self.data)
        self.__getitem__(1)

    def __getitem__(self, index):
        image = tensor(self.images[index])
        masks = tensor(self.masks[index])

        object_ids = torch.unique(masks)[1:]
        number_of_objects = len(object_ids)

        labels = torch.ones((number_of_objects,), dtype=torch.int64)

        bounding_boxes = self.bounding_boxes[0][:][0]
        area = (bounding_boxes[3] - bounding_boxes[1]) * (bounding_boxes[2] - bounding_boxes[0])

        is_crowd = torch.zeros((number_of_objects,), dtype=torch.int64)

        target = {'boxes': tv_tensors.BoundingBoxes(bounding_boxes, format='XYXY',
                                                    canvas_size=image.shape[:2]),
                  'masks': tv_tensors.Mask(masks), 'labels': labels, 'image_id': index, 'area': area,
                  'iscrowd': is_crowd}
        return image, target

    def __len__(self):
        return len(self.images)

    def __get_properties(self, oc_human_data):
        images = []
        image_masks = []
        bounding_boxes = []
        for file in oc_human_data:
            images.append(cv2.imread(os.path.join(self.root, file['file_name'])))
            image_masks.append(self.__get_binary_masks(file=file))
            bounding_boxes.append(self.__get_bounding_boxes(file=file))
        return images, image_masks, bounding_boxes

    @staticmethod
    def __get_binary_masks(file):
        masks = []
        for index, annotation in enumerate(file['annotations']):
            segmentation = annotation['segms']
            if segmentation is not None:
                masks.append(Poly2Mask(segmentation))
        return masks

    @staticmethod
    def __get_bounding_boxes(file):
        bounding_boxes = []
        for annotation in file['annotations']:
            bounding_box = annotation['bbox']
            if bounding_box is not None:
                bounding_boxes.append(bounding_box)
        return bounding_boxes


dataset = OCHumanDataset(image_root='./images/', oc_human=ochuman)

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:100])
dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

number_of_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_of_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, number_of_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.compile()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 5


def get_current_time():
    now = datetime.now()
    now = now.strftime("%b-%d-%Y %H:%M:%S")
    return now


def save_model(current_model, time, directory):
    model_scripted = torch.jit.script(current_model)
    model_scripted.save(directory + '/checkpoint ' + time + '.pt')


filename = "run " + get_current_time()
run_directory = "runs/" + filename
os.mkdir(run_directory)

for epoch in trange(num_epochs, desc='Training Epoch'):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
    save_model(model, get_current_time(), run_directory)
