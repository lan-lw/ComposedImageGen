# Generative Zero-Shot Composed Image Retrieval
<img width="2328" height="344" alt="image" src="https://github.com/user-attachments/assets/b4a3956c-4526-483e-8512-ba518a2b37d8" />

Zero-Shot Composed Image Retrieval vs. Pseudo Target-Aided Composed Image Retrieval. Conventional ZS-CIR methods map the image latent embedding into the token embedding space by textual inversion. The proposed Pseudo Target-Aided method provide additional information for composed embeddings from pseudo-target images.



## Getting Started
### 1.  Installing the dependencies.
```
pip3 install -r requirement.txt
```

### 2. Prepare datasets

#### CIRR

Download [**CIRR dataset**](https://github.com/Cuberick-Orion/CIRR).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```

#### FashionIQ

Download [**FashionIQ dataset**](https://github.com/XiaoxiaoGuo/fashion-iq).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── FashionIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── images
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]
```

#### CIRCO

Download [**CIRCO**](https://github.com/miccunifi/CIRCO).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRCO
│   ├── annotations
|   |   ├── [val | test].json

│   ├── COCO2017_unlabeled
|   |   ├── annotations
|   |   |   ├──  image_info_unlabeled2017.json
|   |   ├── unlabeled2017
|   |   |   ├── [000000243611.jpg | 000000535009.jpg | ...]
```


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
