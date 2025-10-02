# Generative Zero-Shot Composed Image Retrieval
<img width="2328" height="344" alt="image" src="https://github.com/user-attachments/assets/b4a3956c-4526-483e-8512-ba518a2b37d8" />

Zero-Shot Composed Image Retrieval vs. Pseudo Target-Aided Composed Image Retrieval. Conventional ZS-CIR methods map the image latent embedding into the token embedding space by textual inversion. The proposed Pseudo Target-Aided method provide additional information for composed embeddings from pseudo-target images.



## Getting Started
### 1.  Installing the dependencies.
```
pip3 install -r requirement.txt
```

### 2. Prepare datasets

Please refer to [here](https://github.com/miccunifi/SEARLE/tree/main#data-preparation) to prepare the benchmark datasets.

### 3. Pretrained weights

Pretrained Phi models from [lincir](https://github.com/navervision/lincir) and SD models: [link](https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA?usp=drive_link)

## Run the code
### 1. Textual Inversion
> Extracting composed embedding
```
python extract_lincir_feat.py
```

### 2. Composed Image Generation
> Generating composed images with pretrained weights
```
python test_CIG.py
```

### 3. Testing Baseline + CIG
> SEARLE + CIG
```
cd SEARLE_CIG
python src/generate_test_submission.py --submission-name cirr_sdxl_b32 --eval-type searle --dataset cirr --dataset-path /path/to/CIRR --generated-image-dir /path/to/generated_images
```




## 🔥 Updates
- [x] Pretrained weights
- [x] Inference code
- [ ] Support more benchmarks and baselines
- [ ] Train code

## Citation

```
@inproceedings{wang2025CIG,
  title={Generative zero-shot composed image retrieval},
  author={Wang, Lan and Ao, Wei and Boddeti, Vishnu Naresh and Lim, Sernam},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```

## Acknowledgements

This project builds upon the following repositories:

- [SEARLE](https://github.com/miccunifi/SEARLE/tree/main)
- [lincir](https://github.com/navervision/lincir)

I am grateful to the authors and contributors of these projects for making their work available to the community.  
