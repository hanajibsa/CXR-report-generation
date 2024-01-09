from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# import cog
from models.blip import blip_decoder
from tqdm import tqdm

device = "cuda"
# device = 'cpu'
def load_image(image, device='cuda', image_size = 224):
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        #transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

model = blip_decoder('./output/Pretrain/findings_checkpoint_80.pth',image_size=224,vit='base')
model.to(device)
model.eval()
pred_findings = []
gold_findings = []
data_dir = './data/valid_findings_1000.json'

import json
with open(data_dir, 'r') as json_file:
    data = json.load(json_file)

# data = data[:500]
for it in tqdm(data):
    image = it['image']
    text = it['caption']
    gold_findings.append(text)
    im = load_image(image, device=device)
    caption = model.generate(im, sample=True, num_beams=1, max_length=256, min_length=5)
    pred_findings.append(caption)

pred_file = "pred/pred.json"

# Use json.dump() to write the list of strings to the file
with open(pred_file, "w") as outfile:
    for string in pred_findings:
        json_string = json.dumps(string)
        outfile.write(json_string + "\n")

gold_file = "gold/gold.json"

# Use json.dump() to write the list of strings to the file
with open(gold_file, "w") as outfile:
    for string in gold_findings:
        json_string = json.dumps(string)
        outfile.write(json_string + "\n")
