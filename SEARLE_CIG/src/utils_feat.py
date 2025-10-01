from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from data_utils import collate_fn
from phi import Phi

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


@torch.no_grad()
def extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 10) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.to(device)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    return index_features, index_names


def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor, temperature: float) -> torch.Tensor:
    # Based on https://github.com/NVlabs/PALAVRA/blob/main/utils/nv.py
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)

    numerator = torch.exp(torch.diag(torch.inner(v1, v2)) / temperature)
    numerator = torch.cat((numerator, numerator), 0)
    joint_vector = torch.cat((v1, v2), 0)
    pairs_product = torch.exp(torch.mm(joint_vector, joint_vector.t()) / temperature)
    denominator = torch.sum(pairs_product - pairs_product * torch.eye(joint_vector.shape[0]).to(device), 0)

    loss = -torch.mean(torch.log(numerator / denominator))

    return loss

@torch.no_grad()
def normalize_batch(batch):
    norm = torch.linalg.norm(batch, dim=1, keepdim=True)
    norm[norm == 0] = 1
    return batch / norm

@torch.no_grad()
# Function to compute slerp for a batch of tensors
def slerp_batch(v0, v1, t):
    v0 = normalize_batch(v0)
    v1 = normalize_batch(v1)

    dot = torch.sum(v0 * v1, dim=1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)


    sin_theta = torch.sin(theta)
    weight0 = torch.sin((1 - t) * theta) / sin_theta
    weight1 = torch.sin(t * theta) / sin_theta


    weight0[sin_theta == 0] = 1 - t
    weight1[sin_theta == 0] = t

    interpolated = weight0 * v0 + weight1 * v1
    return interpolated
@torch.no_grad()
def extract_pseudo_tokens_with_phi(clip_model: CLIP, phi: Phi, dataset: Dataset,dataset_name="") -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts pseudo tokens from a dataset using a CLIP model and a phi model
    """
    data_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=10, pin_memory=False,
                             collate_fn=collate_fn)

    predicted_tokens = []
    predicted_tokens_gen = []
    predicted_tokens_gen1 = []
    predicted_tokens_gen2 = []
    names_list = []
    print(f"Extracting tokens using phi model")
    for batch in tqdm(data_loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            id = batch.get('pair_id')
            images_ref = batch.get('reference_image')
            images_gen = batch.get('generated_image')

        if names is None:
            names = batch.get('reference_name')
            names_t = batch.get('target_name')


        images_ref = images_ref.to(device)
        images_gen = images_gen.to(device)

        image_features_ref = clip_model.encode_image(images_ref)
        image_features_gen = clip_model.encode_image(images_gen)

        image_features_ref = image_features_ref

        batch_predicted_tokens_ref = phi(image_features_ref)
        batch_predicted_tokens_gen = phi(image_features_gen)

        predicted_tokens.append(batch_predicted_tokens_ref.cpu())
        predicted_tokens_gen.append(batch_predicted_tokens_gen.cpu())

        if dataset_name == 'fashioniq':
            names_list.extend([x+y+str(z.item()) for x,y,z in zip(names,names_t,id)])
        else:
            names_list.extend(id)
    predicted_tokens = torch.vstack(predicted_tokens)
    predicted_tokens_gen = torch.vstack(predicted_tokens_gen)
    predicted_tokens = [predicted_tokens,predicted_tokens_gen]
    return predicted_tokens, names_list


class CustomTensorDataset(Dataset):
    """
    Custom Tensor Dataset which yields image_features and image_names
    """

    def __init__(self, images: torch.Tensor, names: torch.Tensor):
        self.images = images
        self.names = names

    def __getitem__(self, index) -> dict:
        return {'image': self.images[index],
                'image_name': self.names[index]
                }

    def __len__(self):
        return len(self.images)


def get_templates():
    """
    Return a list of templates
    Same templates as in PALAVRA: https://arxiv.org/abs/2204.01694
    """
    return [
        "This is a photo of a {}",
        "This photo contains a {}",
        "A photo of a {}",
        "This is an illustration of a {}",
        "This illustration contains a {}",
        "An illustrations of a {}",
        "This is a sketch of a {}",
        "This sketch contains a {}",
        "A sketch of a {}",
        "This is a diagram of a {}",
        "This diagram contains a {}",
        "A diagram of a {}",
        "A {}",
        "We see a {}",
        "{}",
        "We see a {} in this photo",
        "We see a {} in this image",
        "We see a {} in this illustration",
        "We see a {} photo",
        "We see a {} image",
        "We see a {} illustration",
        "{} photo",
        "{} image",
        "{} illustration",
    ]
