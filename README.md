# Deepfakes Hybrid Defense
This project is an hybrid aproach to defend agains deepfakes, using the implementation of two methods to defend agains deepfake manipulations:
- Artificial fingerprints: embed a fingerprint into an image in order to know where the images comes from
- Antiforgery: embed an invisible disruption into an image to disturb the generator model of deepfakes

## Prerequisites
This project was modified to run on CPU
- Windows
- Python 3.6 or 3.9
- To install the other Python dependencies, run `pip install -r requirements.txt`


## Abstract
In a digital environment saturated with information, the proliferation of technologies such as Generative Adversarial Networks (GANs) has given rise to a growing threat known
as deepfakes. These GANs facilitate the manipulation of image attributes, from hair color to age and gender, allowing key aspects of identity to be altered, which can be used to spread disinformation, impersonate identities and manipulate public perception for malicious or fraudulent purposes. In response to this challenge, this thesis attempts to propose a proactive defense strategy against the misuse of image generation tools applied for unauthorized visual manipulation. The proposal is based on the fusion of two methods: The first method ArtificialGANFingerprints (Yu et al., 2021) that incorporates fingerprints in images to allow their recovery after being transformed by the GAN, and the second method AntiForgery (R. Wang et al., 2022) that introduces imperceptible visual perturbations to degrade the quality of the generated images and obtain unrealistic or artifactual results. These combined methods allow, on the one hand, to track and authenticate the origin of the images and, on the other, to reduce the effectiveness of GANs in generating trustworthy visual content. The results obtained show that this hybrid approach is a promising preventive measure against the proliferation of manipulated and potentially harmful content in digital environments.


## Preparation
## Datasets
- We experiment on one dataset. Download and unzip images into a folder.
- [CelebA Dataset] 10.000 images were used for fingerprint autoencoder training (encoder and decoder).

**CelebA Dataset**

```
bash download.sh celeba
```
More information about the CelebA dataset can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

**StarGAN Model**

```
bash download.sh pretrained-celeba-256x256
```

If you want to train an encoder and decoder from the scratch you can use the next script, or use the pretrained model for 256x256 images.
## Fingerprint autoencoder training
- Run, e.g.,
  ```
  python train_fingerprints.py \
  --data_dir /path/to/images/ \
  --image_resolution 256 \
  --output_dir /path/to/output/ \
  --fingerprint_length 100 \
  --batch_size 4
  ```
  where
  - `image_resolution` indicates the image resolution for training. All the images in `data_dir` is center-cropped according to the shorter side and then resized to this resolution. 
  - `output_dir` contains model snapshots, image snapshots, and log files. For model snapshots, `*_encoder.pth` and `*_decoder.pth` correspond to the fingerprint encoder and decoder respectively.

## Fingerprint embedding and detection
- For **fingerprint embedding**, run, e.g.,
  ```
  python embed_fingerprints.py \
  --encoder_path /path/to/encoder/ \
  --data_dir /path/to/images/ \
  --image_resolution 256 \
  --output_dir /path/to/output/ \
  --identical_fingerprints \
  --batch_size 4
  ```
  where
  - `image_resolution` indicates the image resolution for fingerprint embedding. All the images in `data_dir` is center-cropped according to the shorter side and then resized to this resolution. **It should match the input resolution for the well-trained encoder read from `encoder_path`**.
  - `output_dir` contains embedded fingerprint sequence for each image in `embedded_fingerprints.txt` and fingerprinted images in `fingerprinted_images/`.
  - `identical_fingerprints` needs to be active if and only if all the images need to be fingerprinted with the same fingerprint sequence. 
  
- For **fingerprint detection**, run, e.g.,
  ```
  python detect_fingerprints.py \
  --decoder_path /path/to/decoder/ \
  --data_dir /path/to/fingerprinted/images/ \
  --image_resolution 256 \
  --output_dir /path/to/output/ \
  --batch_size 4
  ```
  where
  - `output_dir` contains detected fingerprint sequence for each image in `detected_fingerprints.txt`.
  - `image_resolution` indicates the image resolution for fingerprint detection. All the images in `data_dir` is center-cropped according to the shorter side and then resized to this resolution. **It should match the input resolution for the well-trained decoder read from `decoder_path`**.
  - You must add a base fingerprint in the `detect_fingerprints.py` file from the `embedded_fingerprints.txt` in order to make the detection method work properly.


## Testing the method

There two possibles pipelines to use the method with StarGAN on the CelebA dataset.
The first:
  1. Embed the fingerprints into the images
  2. Use the adversarial attack
```
python main.py --mode test --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_256/models' --result_dir='./results' --test_iters 200000 --attack_iters 100 --batch_size 1
```
  3. Detect the fingerprints from the images

# Pipeline: org -> fingerprint -> adversarial -> gan
![Pipeline: original-fingerprint-adversarial-gan](https://github.com/kvinstuard/Antiforgery_and_fingerprints/raw/main/assets/escenario5b.png)


The Second (branch/adv-fing):
  1. Use the adversarial attack adding the flag --fingerprint   
```
# Test
python main.py --mode test --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir=stargan_celeba_256\models --result_dir=results --test_iters 200000 --attack_iters 100 --batch_size 1 --fingerprint
```
  2. Detect the fingerprints from the images

# Pipeline: org  -> adversarial -> fingerprint -> gan

![Pipeline: original-fingerprint-adversarial-gan](https://github.com/kvinstuard/Antiforgery_and_fingerprints/raw/main/assets/escenario4b.png)

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


