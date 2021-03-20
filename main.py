from __future__ import print_function
import torch
import torchvision
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import utils
import json 

parser = argparse.ArgumentParser()
parser.add_argument('--model_filename', type=str, default=None)
parser.add_argument('--label_filename', type=str, default=None)
parser.add_argument('--input_image', type=str, required=True)
parser.add_argument('--k_size', type=int, default=23)
parser.add_argument('--thr', type=float, default=0.0)
args = parser.parse_args()

if args.model_filename is not None or args.label_filename is not None:
    assert os.path.exists(args.model_filename)
    assert os.path.exists(args.label_filename)    
assert os.path.exists(args.input_image)
assert args.k_size >= 3 and args.k_size % 2 == 1
assert args.thr >= 0.0

# check cuda support
HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

print('Loading pretrained model')
if args.model_filename is not None:
    model = torch.load(args.model_filename)
    label_file = args.label_filename
else:
    model = torchvision.models.vgg16_bn(pretrained=True)
    #label_file = 'data/ilsvrc_2012_labels.txt'
    label_file = 'imagenet_class_index.json' 
model.eval()
if HAS_CUDA:
    model.cuda()

print('Load and [pre]process image')
target_size = (224, 224)
image_orig = Image.open(args.input_image)
image_orig = image_orig.resize(target_size, Image.NEAREST)
image_orig = np.asarray(image_orig)

mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

image = transf(image_orig).unsqueeze(0)
if HAS_CUDA:
    image = image.cuda()

pred = model(torch.autograd.Variable(image))
pred = pred.data.cpu().numpy().squeeze()

label = json.load(open(label_file))
class_id, class_label, class_prob = utils.predictions_to_class_info(pred, label)
print(class_id, class_label, class_prob)

print('Computing saliency map')
smap = utils.compute_saliency_map(model, image[0], args.k_size, class_id, class_prob, args.thr)

#from scipy.misc import imresize
#mask = imresize(smap, image_orig.shape[:2])
mask = np.array(Image.fromarray(smap).resize(image_orig.shape[:2]))

plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(image_orig)
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(smap, cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.subplot(1,2,2)
plt.title(class_label + ' [{:.1f} %]'.format(100 * class_prob))
plt.imshow(image_orig)
plt.imshow(mask, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.savefig('saliency.svg')
plt.show()
