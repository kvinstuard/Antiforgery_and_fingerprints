import torch
import numpy as np
import argparse
import os
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from color_space import *
from data_loader import get_loader
from utils import *
from model import Generator, Discriminator
from fingerprint_models import StegaStampEncoder
from torchvision import transforms
from PIL import Image

def read_image_as_tensor(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


"""
Test and evaluate a trained model (StarGAN) on the dataset CelebA.
It generate adversarial images, embed a fingerprint, modifying them, and compute several evaluation metrics.
"""

def is_valid_tensor(tensor):
    """
    Check if the tensor is valid (contains no NaN or infinite values).
    """
    return torch.isfinite(tensor).all() and not torch.isnan(tensor).any()

def main():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # Data configuration.
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--attack_iters', type=int, default=100)
    parser.add_argument('--resume_iters', type=int, default=200000, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    # Data paths
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='stargan_celeba_256/models')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--adv_image_dir', type=str, default='results/adv_images')
    parser.add_argument('--modified_image_dir', type=str, default='results/modified_images')
    parser.add_argument('--fingerprint', action='store_true', help='Add watermark to images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save watermarked images')
    parser.add_argument('--fingerprints_file', type=str, default='output/embed_fingerprints_to_adv.txt', help='File to save the fingerprints')


    config = parser.parse_args()

    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.adv_image_dir, exist_ok=True)
    os.makedirs(config.modified_image_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    

    def generate_random_fingerprints(fingerprint_size, batch_size=4):
        z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
        return z
    
    global fingerprints
    encoder_path = "...\CelebA_256x256_encoder.pth"
    state_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    global fingerprint_size
    fingerprint_size = state_dict["secret_dense.weight"].shape[-1] 
    fingerprints = generate_random_fingerprints(fingerprint_size, 1)
    
    def embed_fingerprints_for_gan(images, output_dir, image_resolution=256, identical_fingerprints=True, batch_size=4):
        """
        This function embeds fingerprints in the provided images.
        The input `images` must be an image tensor and return a tensor of images with the fingerprints embedded.
        """
    
        global fingerprints
        global fingerprint_size
        
        # StegaStampEncoder
        HideNet = StegaStampEncoder(image_resolution, 3, fingerprint_size=fingerprint_size, return_residual=False)
        HideNet.load_state_dict(state_dict)
        HideNet.eval()

        # Creating fingerprints
        if identical_fingerprints:
            fingerprints = fingerprints.view(1, fingerprint_size).expand(images.size(0), fingerprint_size)
        else:
            fingerprints = generate_random_fingerprints(fingerprint_size, images.size(0))
        
        # Embedding fingerprints into images
        with torch.no_grad():
            fingerprinted_images = HideNet(fingerprints[:images.size(0)], images)        

        # Saving the fingerprints and the images in the files
        fingerprints_dir = os.path.join(output_dir, "fingerprinted_images")
        os.makedirs(fingerprints_dir, exist_ok=True)
        
        fingerprints_file_path = os.path.join(output_dir, "embedded_fingerprints.txt")
        
        # Openning the file in mode ("append") to avoid losing info every iteration
        with open(fingerprints_file_path, "a") as f:
            for idx, (image, fingerprint) in enumerate(zip(fingerprinted_images, fingerprints)):

                image_filename = f"image_with_fingerprint.jpg"
                image_path = os.path.join(fingerprints_dir, image_filename)
                
                save_image(image, image_path, padding=0)
                
                # Fingerprint to string
                fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
                f.write(f"{image_filename} {fingerprint_str}\n")
                idx += 1
        
        # Returning the images with fingerprints as tensors
        return fingerprinted_images


    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               'CelebA', config.mode, config.num_workers)

    # Load models
    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num)
    D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num)

    print('Loading the trained models from step {}...'.format(config.resume_iters))
    G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(config.resume_iters))
    D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(config.resume_iters))
    
    load_model_weights(G, G_path)
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    print("loading model successful")

    l2_error, ssim, psnr = 0.0, 0.0, 0.0
    n_samples, n_dist = 0, 0

    for i, (x_real, c_org) in enumerate(celeba_loader):
        c_trg_list = create_labels(c_org, config.c_dim, 'CelebA', config.selected_attrs)
        x_fake_list = [x_real]

        x_adv, pert = lab_attack(x_real, c_trg_list, G, iter=config.attack_iters)

        # Save the adversarial image without fingerprint (watermark)
        adv_image_path_no_watermark = os.path.join(config.adv_image_dir, '{}-adv-image-no-watermark.jpg'.format(i + 1))
        save_image(denorm(x_adv.data.cpu()), adv_image_path_no_watermark, nrow=1, padding=0)

        x_adv = read_image_as_tensor(adv_image_path_no_watermark).to(x_real.device)

        # Apply fingerprint (watermark)
        if config.fingerprint:
            x_adv = embed_fingerprints_for_gan(x_adv, 
                                            output_dir=config.output_dir,
                                            image_resolution=256,
                                            identical_fingerprints=True)

        # Save the adversarial image with fingerprint
        adv_image_path_with_watermark = os.path.join(config.adv_image_dir, '{}-adv-image-with-watermark.jpg'.format(i + 1))
        # Avoding denormalization
        save_image(x_adv.data.cpu(), adv_image_path_with_watermark, nrow=1, padding=0)

        
        for idx, c_trg in enumerate(c_trg_list):
            print('image', i, 'class', idx)
            with torch.no_grad():
                x_real_mod = x_real
                gen_noattack, gen_noattack_feats = G(x_real_mod, c_trg)

            # Generate and save modified images
            with torch.no_grad():
                gen, _ = G(x_adv, c_trg)
                
                x_fake_list.append(gen_noattack)
                x_fake_list.append(gen)

                modified_image_path = os.path.join(config.modified_image_dir, '{}-modified-image-class{}.jpg'.format(i + 1, idx))
                save_image(denorm(gen.cpu()), modified_image_path, nrow=1, padding=0)

                # Only calculate metrics if both gen and gen_noattack are valid
                if is_valid_tensor(gen) and is_valid_tensor(gen_noattack):
                    l2_error += F.mse_loss(gen, gen_noattack).item()
                    ssim_local, psnr_local = compare(denorm(gen), denorm(gen_noattack))
                    ssim += ssim_local
                    psnr += psnr_local

                    if F.mse_loss(gen, gen_noattack).item() > 0.05:
                        n_dist += 1
                    n_samples += 1

        x_concat = torch.cat(x_fake_list, dim=3)
        result_path = os.path.join(config.result_dir, '{}-images.jpg'.format(i + 1))
        save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
        
        if i == 100:  # stop after this many images
            break

    # Print metrics
    print('{} images. L2 error: {}. SSIM: {}. PSNR: {}. n_dist: {}'.format(n_samples,
                                                                          l2_error / n_samples,
                                                                          ssim / n_samples,
                                                                          psnr / n_samples,
                                                                          float(n_dist) / n_samples))

if __name__ == '__main__':  
    main()
