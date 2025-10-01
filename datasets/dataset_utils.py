import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import json
from PIL import Image
from torchvision import transforms
import random
from torchvision.transforms.functional import crop
import numpy as np
from PIL import ImageFile


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CIRRDataset(Dataset):
    def __init__(self, dataset_path, split='test', preprocess=None):
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess
        # Load dataset metadata
        with open(os.path.join(dataset_path, f'captions/cap.rc2.{split}.json'), 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        reference_image_path = os.path.join(self.dataset_path, self.split, item['reference']+'.png')
        reference_image = Image.open(reference_image_path).convert('RGB')
        if self.preprocess:
            reference_image = self.preprocess(reference_image)['pixel_values'][0]

        return {
            'reference_image': reference_image,
            'relative_caption': item['caption'],
            'pairid': item['pairid']
        }


class FashionIQDataset(Dataset):
    def __init__(self, dataset_path, split='test', dress_types=['dress', 'shirt', 'toptee'], preprocess=None):
        self.dataset_path = dataset_path
        self.split = split
        self.dress_types = dress_types
        self.preprocess = preprocess
        # Load dataset metadata
        self.metadata = []
        for dress_type in dress_types:
            with open(os.path.join(dataset_path, f'{dress_type}_{split}_metadata.json'), 'r') as f:
                self.metadata.extend(json.load(f))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        reference_image_path = os.path.join(self.dataset_path, 'images', item['reference_image'])
        reference_image = Image.open(reference_image_path).convert('RGB')
        if self.preprocess:
            reference_image = self.preprocess(reference_image)['pixel_values'][0]

        return {
            'reference_image': reference_image,
            'relative_caption': item['relative_caption']
        }


class ComposedEmbedsDataset(Dataset):
    """
    Dataset that loads saved composed embeddings(.pt) .
    Expects files saved as {pairid}.pt containing keys:
      - 'conditioning'  (Tensor [seq_len, d1] or [1, seq_len, d1])
      - 'conditioning2' (Tensor [seq_len, d2] or [1, seq_len, d2])
      - 'pooled2'       (Tensor [d2] or [1, d2])

    Returns for each item:
      {
        'pairid': str,
        'prompt_embeds': Tensor [seq_len, d1+d2],
        'pooled2': Tensor [d2]
      }
    """

    def __init__(self, dataset_path: str, text_embeddings_dir: str, split: str = 'val'):
        self.dataset_path = dataset_path
        self.text_embeddings_dir = text_embeddings_dir
        self.split = split

        # Load caption metadata to get pairids
        cap_path = os.path.join(dataset_path, f'captions/cap.rc2.{split}.json')
        with open(cap_path, 'r') as f:
            metadata = json.load(f)

        # Keep only entries with an existing .pt embedding file
        self.items = []
        for item in metadata:
            pairid = str(item['pairid'])
            pt_path = os.path.join(text_embeddings_dir, f"{pairid}.pt")
            if os.path.exists(pt_path):
                self.items.append({'pairid': pairid, 'pt_path': pt_path})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        pairid = entry['pairid']
        pt_path = entry['pt_path']

        save_dict = torch.load(pt_path, map_location='cpu')

        cond1 = save_dict['conditioning']  # [seq_len, d1] or [1, seq_len, d1]
        cond2 = save_dict['conditioning2'] # [seq_len, d2] or [1, seq_len, d2]
        pooled2 = save_dict['pooled2']     # [d2] or [1, d2]

        # Normalize shapes to [seq_len, d]
        if cond1.dim() == 3:
            cond1 = cond1.squeeze(0)
        if cond2.dim() == 3:
            cond2 = cond2.squeeze(0)
        if pooled2.dim() == 2:
            pooled2 = pooled2.squeeze(0)

        prompt_embeds = torch.concat([cond1, cond2], dim=-1)

        return {
            'pairid': pairid,
            'prompt_embeds': prompt_embeds,  # CPU tensor; caller can .to('cuda')
            'pooled2': pooled2,
        }