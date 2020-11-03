#这是训练完D后的第二步，这里需要训练G,让其能生成对应的真实图片,考虑两类loss，原ganLoss, MSE-Loss
#这还是第二步改良，这里只考虑训练D，这部要把一个真实图像尽可能嵌入embedding中，让G真实的显示

import torch
import numpy as np
import os
import torchvision
from pro_gan_pytorch import  Encoder , Networks as net
from pro_gan_pytorch.DataTools import DatasetFromFolder
from torch.autograd import Variable

#device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------path setting---------------
resultPath = "./result/Step2_Training_G_V1_pic1"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath1_1):
    os.mkdir(resultPath1_1)

resultPath1_2 = resultPath+"/models"
if not os.path.exists(resultPath1_2):
    os.mkdir(resultPath1_2)


#----------------test pre-model output-----------

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 

netE = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
netE.load_state_dict(torch.load('./pre-model/D2E_std_L1_ep9.pth',map_location=device))


#-------------load single image--------------
loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

from PIL import Image
def image_loader(image_name):
	image = Image.open(image_name).convert('RGB')
	image = image.resize((1024,1024))
	image = loader(image).unsqueeze(0)
	return image.to(torch.float)

im1=image_loader('./1.jpg')

im1 = im1*2-1

# --------------training with generative image------------
optimizer = torch.optim.Adam(netE.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
loss = torch.nn.MSELoss()
loss_all=0
for epoch in range(10):
	for i in range(5001):
		im1 = im1.to(device)
		z = netE(im1.detach(),height=8,alpha=1)
		z = z.squeeze(2).squeeze(2)
		x = netG(z,depth=8,alpha=1)
		optimizer.zero_grad()
		loss_i = loss(x,im1)
		loss_i.backward()
		optimizer.step()
		loss_all +=loss_i.item()
		print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
		if (i % 100==0) or (i<20 and epoch==0) :
			img = (torch.cat((im1,x))+1)/2 
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=2)
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()),file=f)
			with open(resultPath+'/D_z.txt', 'a+') as f:
				print('D_z:  '+str(z[0,0:30])+'     D_z:    '+str(z[0,30:60]),file=f)
	#if epoch%10==0 or epoch == 29:
	torch.save(netE.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)






















