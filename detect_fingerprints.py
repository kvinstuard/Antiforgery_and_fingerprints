import argparse
import glob
import PIL
import os
from time import time
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from statistics import mean, median, mode

"""
Detects the presence of fingerprints in images from a specified directory.
It uses a trained StegaStamp decoder to extract fingerprints and compares them to a pre-defined base fingerprint.
Processes the images in batches, calculates similarity scores for each image, and saves the results
in a text file, along with the similarity score for each image and whether the watermark is detected above a 
specified threshold. It also prints statistical measures (mean, median, mode, standard deviation) of the 
similarity scores for the entire dataset.
"""



parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument("--output_dir", type=str, help="Path to save watermarked images to.")
parser.add_argument("--image_resolution", type=int, required=True, help="Height and width of square images.")
parser.add_argument("--decoder_path", type=str, required=True, help="Path to trained StegaStamp decoder.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=-1) 
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--similarity_threshold", type=float, default=0.9, help="Required similarity threshold as a decimal.")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

device = torch.device("cuda:0" if args.cuda != -1 and torch.cuda.is_available() else "cpu")

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

def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from fingerprint_models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path, map_location=torch.device('cpu'))
    FINGERPRINT_SIZE = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(args.image_resolution, 3, FINGERPRINT_SIZE)
    RevealNet.load_state_dict(state_dict, strict=False)
    RevealNet = RevealNet.to(device)

def load_data():
    global dataset, dataloader

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
    ])
    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

def get_base_fingerprint():
    base_fingerprint_str = "0100010001000010111010111111110011101000001111101101010110000000011011110101110101010010111111111110"
    return torch.tensor([int(bit) for bit in base_fingerprint_str], dtype=torch.long)

def calculate_similarity(fingerprint1, fingerprint2):
    return (fingerprint1 == fingerprint2).sum().item() / len(fingerprint1)

def extract_fingerprints(base_fingerprint, threshold):
    all_fingerprinted_images = []
    all_fingerprints = []
    similarity_scores = []

    BATCH_SIZE = args.batch_size
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for images, _ in tqdm(dataloader):
        images = images.to(device)

        fingerprints = RevealNet(images)
        fingerprints = (fingerprints > 0).long()
        
        all_fingerprinted_images.append(images.detach().cpu())
        all_fingerprints.append(fingerprints.detach().cpu())

    dirname = args.output_dir
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "detected_fingerprints.txt"), "w")
    for idx in range(len(all_fingerprints)):
        fingerprint = all_fingerprints[idx]
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        similarity = calculate_similarity(fingerprint, base_fingerprint)
        
        similarity_scores.append(similarity)

        if similarity > threshold:
            _, filename = os.path.split(dataset.filenames[idx])
            filename = filename.split('.')[0] + ".jpg"
            f.write(f"{filename} {fingerprint_str} {similarity:.2f}\n")
        else:
            _, filename = os.path.split(dataset.filenames[idx])
            filename = filename.split('.')[0] + ".jpg"
            f.write(f"{filename} {fingerprint_str} {similarity:.2f} - No fingerprint detected\n")
            print("No contiene un fingerprint") 
    f.close()


    mean_score = mean(similarity_scores)
    median_score = median(similarity_scores)
    mode_score = mode(similarity_scores)
    std_deviation = np.std(similarity_scores)

    print("\n=== Estadísticas de Similitud ===")
    print(f"Promedio: {mean_score:.4f}")
    print(f"Mediana: {median_score:.4f}")
    print(f"Moda: {mode_score:.4f}")
    print(f"Desviación Estándar: {std_deviation:.4f}")

if __name__ == "__main__":
    load_decoder()
    load_data()
    base_fingerprint = get_base_fingerprint()
    extract_fingerprints(base_fingerprint, args.similarity_threshold)