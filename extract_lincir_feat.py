# Imports
import os
import torch
import clip
from PIL import Image, ImageFile
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
from phi import Phi
from datasets.dataset_utils import CIRRDataset, FashionIQDataset
import argparse
from tqdm.auto import tqdm

# PIL image settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

# Argument parser for dataset selection
parser = argparse.ArgumentParser(description="Choose dataset to load")
parser.add_argument('--dataset', type=str, choices=['cirr', 'fashioniq'], default='cirr',
                    help="Specify the dataset to load: 'cirr' or 'fashioniq'")
parser.add_argument('--text_embeddings_dir', type=str,
                    default="/path/to/embedded_text_lincir_cirr_test/",
                    help="Directory to save extracted embeddings (.pt files)")
args = parser.parse_args()

text_embeddings_dir = args.text_embeddings_dir
os.makedirs(text_embeddings_dir, exist_ok=True)

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Function to encode text with pseudo tokens
def encode_with_pseudo_tokens_HF(clip_model: CLIPTextModelWithProjection, text: torch.Tensor, pseudo_tokens: torch.Tensor, num_tokens=1, return_last_states=True) -> torch.Tensor:
    """
    Encode text with pseudo tokens using HuggingFace CLIP model.
    """
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)
    x = torch.where(text.unsqueeze(-1) == 259, pseudo_tokens.unsqueeze(1).type(clip_model.dtype), x)
    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    x = clip_model.text_model.encoder(
        inputs_embeds=x,
        attention_mask=None,
        causal_attention_mask=_causal_attention_mask,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=False
    )
    prompt_embeds = x[1][-2]
    x = x[0]
    x_last = clip_model.text_model.final_layer_norm(x)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device), text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1)]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)
    if return_last_states:
        return x, x_last, prompt_embeds
    else:
        return x

# Model and processor setup
phi_path = '/path/to/phi_best.pt'
phi_path_2 = '/path/to/phi_best.pt'
clip_model_name = 'openai/clip-vit-large-patch14'
clip_model_name2 = 'Geonmo/CLIP-Giga-config-fixed'

clip_preprocess = CLIPImageProcessor(
    crop_size={'height': 224, 'width': 224},
    do_center_crop=True,
    do_convert_rgb=True,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
    resample=3,
    size={'shortest_edge': 224},
)

# Load CLIP models
clip_text_model1 = CLIPTextModelWithProjection.from_pretrained(clip_model_name, torch_dtype=torch.float32, cache_dir='./').float().to("cuda")
clip_vision_model1 = CLIPVisionModelWithProjection.from_pretrained(clip_model_name, torch_dtype=torch.float32, cache_dir='./').float().to("cuda")
clip_text_model2 = CLIPTextModelWithProjection.from_pretrained(clip_model_name2, torch_dtype=torch.float32, cache_dir='./').float().to("cuda")
clip_vision_model2 = CLIPVisionModelWithProjection.from_pretrained(clip_model_name2, torch_dtype=torch.float32, cache_dir='./').float().to("cuda")

# Load Phi models
phi = Phi(input_dim=768, hidden_dim=768 * 4, output_dim=768, dropout=0)
phi_2 = Phi(input_dim=1280, hidden_dim=1280 * 4, output_dim=1280, dropout=0)
phi.load_state_dict(torch.load(phi_path, map_location="cuda")[phi.__class__.__name__])
phi_2.load_state_dict(torch.load(phi_path_2, map_location="cuda")[phi.__class__.__name__])
phi = phi.to(device="cuda").eval()
phi_2 = phi_2.to(device="cuda").eval()

# Dataset selection
if args.dataset == 'cirr':
    dataset_path = "/path/to/cirr/dataset"
    dataset = CIRRDataset(dataset_path, split='test1', preprocess=clip_preprocess)
elif args.dataset == 'fashioniq':
    dataset_path = "/path/to/fashioniq/dataset"
    dataset = FashionIQDataset(dataset_path, split='test', dress_types=['dress', 'shirt', 'toptee'], preprocess=clip_preprocess)

# DataLoader setup
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
)

# Main loop for extracting and saving features
for batch in tqdm(dataloader, desc=f"Extracting features ({args.dataset})", unit="batch"):
    pairid = batch.get('pairid', None)  # Use 'id' if available, else None
    reference_image = batch['reference_image']

    # Extract image features
    image_features1 = clip_vision_model1(reference_image.to("cuda")).image_embeds
    image_features2 = clip_vision_model2(reference_image.to("cuda")).image_embeds

    # Predict pseudo tokens
    predicted_tokens = phi(image_features1.to(torch.float32))
    predicted_tokens2 = phi_2(image_features2.to(torch.float32))

    # Prepare input captions
    relative_captions = batch.get('relative_caption', ["No caption"] * len(reference_image))
    input_captions = [f"a photo of $ that {cap}" for cap in relative_captions]
    tokenized_input_captions = clip.tokenize(input_captions, context_length=77, truncate=True).to("cuda")
    tokenized_input_captions2 = clip.tokenize(input_captions, context_length=77, truncate=True).to("cuda")

    # Encode with pseudo tokens
    _, pooled, conditioning = encode_with_pseudo_tokens_HF(clip_text_model1, tokenized_input_captions, predicted_tokens)
    pooled2, _, conditioning2 = encode_with_pseudo_tokens_HF(clip_text_model2, tokenized_input_captions2, predicted_tokens2)

    # Save results
    for idx in range(len(reference_image)):
        save_dict = {
            'pooled': pooled[idx].cpu().data,
            'conditioning': conditioning[idx].cpu().data,
            'pooled2': pooled2[idx].cpu().data,
            'conditioning2': conditioning2[idx].cpu().data
        }

        torch.save(save_dict, os.path.join(text_embeddings_dir, f"{pairid[idx]}.pt"))
       