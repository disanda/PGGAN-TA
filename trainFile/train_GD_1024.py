#这是训练完D后的第二步，这里需要训练G,让其能生成对应的真实图片,考虑两类loss，原ganLoss, MSE-Loss
#控制编码，让其和分布接近

import torch
import numpy as np
import os
import torchvision
from pro_gan_pytorch import  Encoder , Networks as net
from pro_gan_pytorch.DataTools import DatasetFromFolder
from torch.autograd import Variable
import itertools

#device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------path setting---------------
resultPath = "./result/Step2_Training_EG_V1"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath1_1):
    os.mkdir(resultPath1_1)

resultPath1_2 = resultPath+"/models"
if not os.path.exists(resultPath1_2):
    os.mkdir(resultPath1_2)

#-----------------test preModel-------------------
#print(gen)
# depth=0
# z = torch.randn(4,512)
# x = (gen1(z,depth=depth,alpha=1)+1)/2
# torchvision.utils.save_image(x, './face_dp%d.jpg'%depth, nrow=4)
# del x
# x = (gen2(z,depth=depth,alpha=1)+1)/2
# torchvision.utils.save_image(x, './face-shadow%d.jpg'%depth, nrow=4)

#---------test output------------
# netD = Encoder.encoder_v1(height=9, feature_size=512)

# #print(netD.final_block)
# x = torch.randn(1,3,1024,1024)
# z = netD(x,height=8,alpha=1)
# print(z.shape)

# netG = Networks.Generator(depth=9, latent_size=512)
# z = z.squeeze(2).squeeze(2)
# x_ = netG(z,depth=8,alpha=1)
# print(z.shape)
# print(x_.shape)


#----------------test pre-model output-----------

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
netD1 = torch.nn.DataParallel(net.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netD1.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))

netD2 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
#netD2.load_state_dict(torch.load('./pre-model/AG_E_model_ep9.pth',map_location=device))
#netD2 = torch.nn.DataParallel(Encoder.encoder_v2()) #新结构，不需要参数 
toggle_grad(netD1,False)
toggle_grad(netD2,False)

paraDict = dict(netD1.named_parameters()) # pre_model weight dict
for i,j in netD2.named_parameters():
	if i in paraDict.keys():
		w = paraDict[i]
		j.copy_(w)

toggle_grad(netD2,True)

del netD1

#---------------training with true image-------------
# optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
# loss = torch.nn.MSELoss()
# loss_all=0
# for epoch in range(10):
# 	for (i, batch) in enumerate(data):
# 		image = batch.to(device)
# 		z = netD2(image,height=8,alpha=1)
# 		z = z.squeeze(2).squeeze(2)
# 		x_ = netG(z.detach(),depth=8,alpha=1) #这个去梯度，会没有效果, (训练结果基本不会发生改变)!
# 		optimizer.zero_grad()
# 		loss_i = loss(x_,image)
# 		loss_i.backward()
# 		optimizer.step()
# 		loss_all +=loss_i.item()
# 		if i % 100 == 0:
# 			print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
# 			x_ = (x_+1)/2
# 			img = torch.cat((image[:8],x_[:8]))
# 			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
# 	torch.save(netG.state_dict(), resultPath1_2+'/G_model.pth')
# 	torch.save(netD2.state_dict(), resultPath1_2+'/D_model.pth')

#---------------training with true image & noise-------------
# optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
# loss = torch.nn.MSELoss()
# #loss = torch.nn.BCELoss()
# loss_all=0
# for epoch in range(10):
# 	for (i, batch) in enumerate(data):
# 		#z_ = torch.randn(10, 512).to(device)
# 		image = batch.to(device)
# 		z = netD2(image,height=8,alpha=1)
# 		z = z.squeeze(2).squeeze(2)
# 		x_ = netG(z,depth=8,alpha=1) #A.这个去梯度，会没有效果, (训练结果基本不会发生改变)! B.用detach,G的梯度不受影响，也影响不到D,人脸不改变，但属性会跟着变 C.什么都不用，G会受当次影响发生改变,生成效果变化比较大
# 		optimizer.zero_grad()
# 		loss_i = loss(x_,image)
# 		loss_i.backward()
# 		optimizer.step()
# 		loss_all +=loss_i.item()
# 		if i % 100 == 0:
# 			print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
# 			#x_ = (x_+1)/2
# 			img = torch.cat((image[:8],x_[:8]))
# 			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
# 	torch.save(netG.state_dict(), resultPath1_2+'/G_model.pth')
# 	torch.save(netD2.state_dict(), resultPath1_2+'/D_model.pth')


#---------------training with true image & compare z------------- 这个完全没有生成model collapse
# optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
# loss = torch.nn.MSELoss()
# loss_all=0
# for epoch in range(10):
# 	for (i, batch) in enumerate(data):
# 		image = batch.to(device)
# 		z = netD2(image,height=8,alpha=1)
# 		z = z.squeeze(2).squeeze(2)
# 		with torch.no_grad():
# 			x = netG(z,depth=8,alpha=1)
# 		z_ = netD2(x,height=8,alpha=1)
# 		z_ = z_.squeeze(2).squeeze(2)
# 		optimizer.zero_grad()
# 		loss_i = loss(z,z_)
# 		loss_i.backward()
# 		optimizer.step()
# 		loss_all +=loss_i.item()
# 		if i % 100 == 0: 
# 			print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
# 			img = (torch.cat((image[:8],x[:8]))+1)/2
# 			torchvision.utils.save_image(image, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
# 	if epoch%10==0 or epoch == 29:
# 		torch.save(netG.state_dict(), resultPath1_2+'/G_model.pth')
# 		torch.save(netD2.state_dict(), resultPath1_2+'/D_model.pth')



#-------------load single image--------------
# loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# from PIL import Image
# def image_loader(image_name):
# 	image = Image.open(image_name).convert('RGB')
# 	image = image.resize((1024,1024))
# 	image = loader(image).unsqueeze(0)
# 	return image.to(torch.float)

# im1=image_loader('./1.jpg')


# --------------training with generative image------------
import functools
#optimizer = torch.optim.Adam(netG.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
optimizer = torch.optim.Adam(itertools.chain(netG.parameters(), netD2.parameters()),lr=0.001 ,betas=(0, 0.99), eps=1e-8)
loss = torch.nn.MSELoss()
#loss = torch.nn.BCEloss()
loss_all=0
for epoch in range(10):
	for i in range(5001):
		z = torch.randn(5, 512).to(device)
		x = netG(z,depth=8,alpha=1)
		z_ = netD2(x.detach(),height=8,alpha=1)
		z_ = z_.squeeze(2).squeeze(2)
		x_ = netG(z_,depth=8,alpha=1)
		optimizer.zero_grad()
		loss_1 = loss(x,x_)
		loss_2 = loss(z.mean(),z_.mean())
		loss_i =loss_1 + loss_2
		loss_i.backward()
		optimizer.step()
		loss_all +=loss_i.item()
		print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
		if (i % 100==0) or (i<20 and epoch==0) : 
			img = (torch.cat((x[:5],x_[:5]))+1)/2
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=5)
			#torchvision.utils.save_image(x_[:8], resultPath1_1+'/%d_rc.jpg'%(epoch,i), nrow=8)
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print(str(epoch)+'-'+str(i)+'-'+'loss_all__:'+str(loss_all)+'loss_1:'+str(loss_1.item())+'loss_2:'+str(loss_2.item()),file=f)
			with open(resultPath+'/D_z.txt', 'a+') as f:
				print(str(epoch)+'-'+str(i)+'-'+'D_z:  '+str(z_[0,0:30])+'     D_z:    '+str(z_[0,30:60]),file=f)
				print(str(epoch)+'-'+str(i)+'-'+'D_z_mean:  '+str(z_.mean())+'     D_z_std:    '+str(z_.std()),file=f)
	#if epoch%10==0 or epoch == 29:
	torch.save(netG.state_dict(), resultPath1_2+'/G_model_ep%d.pth'%epoch)
	#torch.save(netD2.state_dict(), resultPath1_2+'/D_model_ep%d.pth'%epoch)


#--------------training with generative image------------: training G with D
# toggle_grad(netG,False)#关闭netG的梯度
# optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
# loss = torch.nn.MSELoss()
# loss_all=0
# for epoch in range(10):
# 	for i in range(5001):
# 		z = torch.randn(10, 512).to(device)
# 		x = netG(z,depth=8,alpha=1)
# 		z_ = netD2(x.detach(),height=8,alpha=1)
# 		#z_ = netD2(x.detach()) #new_small_Net
# 		z_ = z_.squeeze(2).squeeze(2)
# 		x_ = netG(z_,depth=8,alpha=1)
# 		optimizer.zero_grad()
# 		loss_i = loss(x_,x)
# 		loss_i.backward()
# 		optimizer.step()
# 		loss_all +=loss_i.item()
# 		print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
# 		if i % 100 == 0: 
# 			img = (torch.cat((x[:8],x_[:8]))+1)/2
# 			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
# 			#torchvision.utils.save_image(x_[:8], resultPath1_1+'/%d_rc.jpg'%(epoch,i), nrow=8)
# 	#if epoch%10==0 or epoch == 29:
# 	torch.save(netG.state_dict(), resultPath1_2+'/G_model_ep%d.pth'%epoch)
# 	torch.save(netD2.state_dict(), resultPath1_2+'/D_model_ep%d.pth'%epoch)




















