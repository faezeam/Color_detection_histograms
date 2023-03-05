
from RGBuvHistBlock import RGBuvHistBlock
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from os import listdir
from glob import glob
from os.path import isfile, join

#directory for images dataset
datas=glob("sample/*")

#directory for output numpy histograms
output="Hists/"




def hist_interpolation(hist1, hist2):
  	ratio = torch.rand(1)
  	return hist1 * ratio + hist2 * (1 - ratio)


torch.cuda.set_device(0)

histblock = RGBuvHistBlock(insz=250, h=64,
                           resizing='sampling',
                           method='inverse-quadratic',
                           sigma=0.02,
                           device=torch.cuda.current_device())
transform = transforms.Compose([transforms.ToTensor()])



for data in datas:
	colors=glob(data+"/*")
	os.makedirs(output+data.split('/')[-1])
	for color in colors:
		print(color)
		os.makedirs(output+color.split("/")[1]+"/"+color.split("/")[2])
		files = [join(color, f) for f in listdir(color) if isfile(join(color, f))]
		first = True
		for f in files:
  			first = True
  			img_hist = Image.open(f)
  			file_name=f.split("/")[-1]
  			name=file_name.split(".")[0]
  			img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(
    			device=torch.cuda.current_device())
  			h = histblock(img_hist)
  		
  			histograms = h.cpu().numpy()
  			np.save(output+color.split("/")[1]+"/"+color.split("/")[2]+"/"+name+'.npy', histograms)

