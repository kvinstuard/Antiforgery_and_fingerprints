# Deepfake Proactive Defense
This project is using the implementation of two methods to defend agains deepfake manipulations:
- Artificial fingerprints: embed a fingerprint into an image in order to know where the images come from
- Antiforgery: embed an invisible disruption into an image to disturb the generator model of deepfakes

## Prerequisites
This project was modified to run on CPU
- Windows
- Python 3.6 or 3.9
- To install the other Python dependencies, run `pip install -r requirements.txt`


# Artificial GAN Fingerprints
The complete information can be found in the following links:
(This is a modified version of the README.md file) 
This project was modified to make it work with 256x256 images.

### [Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data](https://arxiv.org/pdf/2007.08457.pdf)
[Ning Yu](https://ningyu1991.github.io/)\*, [Vladislav Skripniuk](https://www.linkedin.com/in/vladislav-skripniuk-8a8891143/?originalSubdomain=ru)\*, [Sahar Abdelnabi](https://s-abdelnabi.github.io/), [Mario Fritz](https://cispa.saarland/group/fritz/)<br>
*Equal contribution<br>
ICCV 2021 Oral

### [paper](https://arxiv.org/pdf/2007.08457.pdf) | [project](https://ningyu1991.github.io/projects/ArtificialGANFingerprints.html) | [poster](https://ningyu1991.github.io/homepage_files/poster_ArtificialGANFingerprints.pdf) | [video](https://www.youtube.com/watch?v=j8bcOHhu4Lg&t=12s)


## Abstract


  
## Datasets
- We experiment on one datasets. Download and unzip images into a folder (newdownload.sh).
  - [CelebA Dataset] 10.000 images for fingerprint autoencoder training (encoder and decoder).
  
## Fingerprint autoencoder training
- Run, e.g.,
  ```
  python train_fingerprints.py \
  --data_dir /path/to/images/ \
  --use_celeba_preprocessing \
  --image_resolution 256 \
  --output_dir /path/to/output/ \
  --fingerprint_length 100 \
  --batch_size 64
  ```
  where
  - `use_celeba_preprocessing` needs to be active if and only if using CelebA aligned and cropped images.
  - `image_resolution` indicates the image resolution for training. All the images in `data_dir` is center-cropped according to the shorter side and then resized to this resolution. When `use_celeba_preprocessing` is active, `image_resolution` has to be set as 128.
  - `output_dir` contains model snapshots, image snapshots, and log files. For model snapshots, `*_encoder.pth` and `*_decoder.pth` correspond to the fingerprint encoder and decoder respectively.



## Fingerprint embedding and detection
- For **fingerprint embedding**, run, e.g.,
  ```
  python embed_fingerprints.py \
  --encoder_path /path/to/encoder/ \
  --data_dir /path/to/images/ \
  --use_celeba_preprocessing \
  --image_resolution 128 \
  --output_dir /path/to/output/ \
  --identical_fingerprints \
  --batch_size 64
  ```
  where
  - `use_celeba_preprocessing` needs to be active if and only if using CelebA aligned and cropped images.
  - `image_resolution` indicates the image resolution for fingerprint embedding. All the images in `data_dir` is center-cropped according to the shorter side and then resized to this resolution. **It should match the input resolution for the well-trained encoder read from `encoder_path`**. When `use_celeba_preprocessing` is active, `image_resolution` has to be set as 128.
  - `output_dir` contains embedded fingerprint sequence for each image in `embedded_fingerprints.txt` and fingerprinted images in `fingerprinted_images/`.
  - `identical_fingerprints` needs to be active if and only if all the images need to be fingerprinted with the same fingerprint sequence. 
  
- For **fingerprint detection**, run, e.g.,
  ```
  python detect_fingerprints.py \
  --decoder_path /path/to/decoder/ \
  --data_dir /path/to/fingerprinted/images/ \
  --image_resolution 128 \
  --output_dir /path/to/output/ \
  --batch_size 64
  ```
  where
  - `output_dir` contains detected fingerprint sequence for each image in `detected_fingerprints.txt`.
  - `image_resolution` indicates the image resolution for fingerprint detection. All the images in `data_dir` is center-cropped according to the shorter side and then resized to this resolution. **It should match the input resolution for the well-trained decoder read from `decoder_path`**.



# Anti-Forgery
An example of **[Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations](https://arxiv.org/abs/2206.00477)** (to be presented at the **IJCAI-ECAI 2022**). This repository contains code for crafting perceptual-aware perturbation in the Lab color space to attack an image-to-image translation network. 

## Preparation
**CelebA Dataset**

```
bash newdownload.sh celeba
```
**StarGAN Model**

```
bash newdownload.sh pretrained-celeba-256x256
```

More information about the CelebA dataset can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

## Attack Testing

Here is a simple example of  testing our method to attack StarGAN on the CelebA dataset.
```
# Test
python main.py --mode test --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_256/models' --result_dir='./results' --test_iters 200000 --attack_iters 100 --batch_size 1
```



## Citation
  ```
  @inproceedings{yu2021artificial,
    author={Yu, Ning and Skripniuk, Vladislav and Abdelnabi, Sahar and Fritz, Mario},
    title={Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data},
    booktitle = {IEEE International Conference on Computer Vision (ICCV)},
    year={2021}
  }
  ```
  ```
@article{wang2022anti,
  title={Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations},
  author={Wang, Run and Huang, Ziheng and Chen, Zhikai and Liu, Li and Chen, Jing and Wang, Lina},
  journal={arXiv preprint arXiv:2206.00477},
  year={2022}
}

```


