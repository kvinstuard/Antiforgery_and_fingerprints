import argparse
import os
import glob
import PIL


"""
Embeds secret fingerprints into images using a trained StegaStamp encoder and decoder.
It supports two modes: using identical fingerprints for all images or randomly generated fingerprints for each image.
The images are processed with optional preprocessing for CelebA dataset (if specified). Performs the following:
    1. Loads the images from a specified directory and applies preprocessing.
    2. Loads the trained encoder and decoder models.
    3. Embeds the fingerprints into the images using the encoder.
    4. Saves the watermarked images along with their associated fingerprints and generates visual samples of clean, watermarked, and residual images.
The results, including the embedded fingerprints and accuracy information, are saved to a specified output directory.
"""


parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--encoder_path", type=str, help="Path to trained StegaStamp encoder.")
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument("--output_dir", type=str, help="Path to save watermarked images to.")
parser.add_argument("--image_resolution", type=int, help="Height and width of square images.")
parser.add_argument("--identical_fingerprints", action="store_true", help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints.")
parser.add_argument("--check", action="store_true", help="Validate fingerprint detection accuracy.")
parser.add_argument("--decoder_path", type=str, help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

def load_data():
    global dataset, dataloader

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

def load_models():
    global HideNet, RevealNet
    global FINGERPRINT_SIZE
    
    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from fingerprint_models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path, map_location=torch.device('cpu'))
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )

    kwargs = {"map_location": "cpu"}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    # generate identical fingerprints
    fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
    fingerprints = fingerprints.view(1, FINGERPRINT_SIZE).expand(BATCH_SIZE, FINGERPRINT_SIZE)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)

    bitwise_accuracy = 0

    for images, _ in tqdm(dataloader):

        # generate arbitrary fingerprints
        if not args.identical_fingerprints:
            fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE)
            fingerprints = fingerprints.view(BATCH_SIZE, FINGERPRINT_SIZE)

        fingerprinted_images = HideNet(fingerprints[: images.size(0)], images)
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())

        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()

    dirname = args.output_dir
    if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
        os.makedirs(os.path.join(dirname, "fingerprinted_images"))

    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    for idx in range(len(all_fingerprinted_images)):
        image = all_fingerprinted_images[idx]
        fingerprint = all_fingerprints[idx]
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".jpg" #change
        save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{filename}"), padding=0)
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        f.write(f"{filename} {fingerprint_str}\n")
    f.close()

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
        print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")

        save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        save_image(torch.abs(images - fingerprinted_images)[:49], os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)

def main():
    load_data()
    load_models()
    embed_fingerprints()

if __name__ == "__main__":
    main()
