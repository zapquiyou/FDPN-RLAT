import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
import torch
import torch.nn as nn
from glob import glob
import os.path as osp
import jsonlines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SceneFlowDataset(Dataset):
    def __init__(self, root, dstype, transform=None):
        self.root = root
        self.dstype = dstype
        self.left_images = sorted(glob(osp.join(root, dstype, 'flying', '*/*/*/left/*.png'))) + sorted(glob(osp.join(root, dstype, 'monkaa', '*/left/*.png'))) + sorted(glob(osp.join(root, dstype, 'driving', '*/*/*/left/*.png')))
        self.right_images = [img.replace('left', 'right') for img in self.left_images]
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):        
        left_image = Image.open(self.left_images[idx]).convert("RGB")
        right_image = Image.open(self.right_images[idx]).convert("RGB")

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        return left_image, right_image

model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name).to(device)
vit_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ViTFusionModel(nn.Module):
    def __init__(self, vit_model):
        super(ViTFusionModel, self).__init__()
        self.vit_model = vit_model

    def forward(self, left_image, right_image):
        left_features = self.vit_model(pixel_values=left_image).last_hidden_state
        right_features = self.vit_model(pixel_values=right_image).last_hidden_state

        fused_features = torch.cat((left_features.mean(dim=1), right_features.mean(dim=1)), dim=-1)
        
        return fused_features

model = ViTFusionModel(vit_model).to(device)

root_list = ['/data/StereoDatasets/sceneflow_rltest/', '/data/StereoDatasets/sceneflow_rltrain/']
dstype='frames_finalpass'
output_file = ['/data/StereoDatasets/sceneflow_vit_fusion_features_rltest.jsonl', '/data/StereoDatasets/sceneflow_vit_fusion_features_rltrain.jsonl']

for root, output in zip(root_list, output_file):
    with jsonlines.open(output, mode='w') as writer:
        dataset = SceneFlowDataset(root, dstype, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for idx, (left_image, right_image) in enumerate(dataloader):
                print(f"Processing image pair {idx + 1}/{len(dataset)}: {dataset.left_images[idx]} and {dataset.right_images[idx]}")
                left_image_path = dataset.left_images[idx]
                right_image_path = dataset.right_images[idx]

                left_image = left_image.to(device)
                right_image = right_image.to(device)

                fused_features = model(left_image, right_image).cpu().numpy()

                record = {
                    "left_image_path": left_image_path,
                    "right_image_path": right_image_path,
                    "fused_features": fused_features.tolist()
                }

                writer.write(record)
