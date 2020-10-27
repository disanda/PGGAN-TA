#这个版本只需要导入网络即可(不需要导入训练网络)，先已完成两个实验，第一个实验完成gt编码的比较，第二个实验完成G(z)的编码比较
#准备做 不同网络的比较，包括结构不同，weight不同的情况 (mnist中以上因素不同，区别不大)
import torch
import torch.nn as nn
import numpy as np
import os
import torchvision
from pro_gan_pytorch import  Encoder , Networks as net1
import pro_gan_pytorch.AE as net2
from pro_gan_pytorch.DataTools import DatasetFromFolder
import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #----------------path setting---------------
resultPath = "./result/RC_Training_GD_V2_1"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath1_1):
    os.mkdir(resultPath1_1)

resultPath1_2 = resultPath+"/models"
if not os.path.exists(resultPath1_2):
    os.mkdir(resultPath1_2)



#----------------pre-model-----------

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

in_dim = 512

netG1 = torch.nn.DataParallel(net1.Generator(depth=9,latent_size=in_dim))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netG1.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
netD1 = torch.nn.DataParallel(net1.Discriminator(height=9, feature_size=in_dim))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netD1.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))

netG2 = torch.nn.DataParallel(net2.Decoder_v1(depth=9,latent_size=in_dim))
netD2 = torch.nn.DataParallel(net2.Encoder_v1(height=9, feature_size=in_dim)) #新结构，不需要参数 

toggle_grad(netD1,False)
toggle_grad(netD2,False)
paraDict = dict(netD1.named_parameters()) # pre_model weight dict
for i,j in netD2.named_parameters():
	if i in paraDict.keys():
		w = paraDict[i]
		j.copy_(w)
	else:
		j.requires_grad_(True)


toggle_grad(netG1,False)
toggle_grad(netG2,False)
paraDict = dict(netG1.named_parameters())
for i,j in netG2.named_parameters():
	if i in paraDict.keys():
		w = paraDict[i]
		j.copy_(w)
	else:
		j.requires_grad_(True)

optimizerG = torch.optim.Adam(netG2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8,amsgrad=True)
optimizerD = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8,amsgrad=True)

with open(resultPath+'/setting.txt', 'w') as f:
	print('----',file=f)
	print(netG1,file=f)
	print('----',file=f)
	print(netD1,file=f)
	print('----',file=f)
	print(netG2,file=f)
	print('----',file=f)
	print(netD2,file=f)
	print('----',file=f)
	print(optimizerG,file=f)
	print('----',file=f)
	print(optimizerD,file=f)
	print('----',file=f)


#------------------dataSet-----------
# data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
# #data_path='/Users/apple/Desktop/CelebAMask-HQ/CelebA-HQ-img'
# trans = torchvision.transforms.ToTensor()
# dataSet = DatasetFromFolder(data_path,transform=trans)
# data = torch.utils.data.DataLoader(dataset=dataSet,batch_size=10,shuffle=True,num_workers=4,pin_memory=True)

# image = next(iter(data))
# torchvision.utils.save_image(image, './1.jpg', nrow=1)


#--------------training with generative image------------: training G with D
#optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
import itertools
#optimizer = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netG2.parameters()), filter(lambda p: p.requires_grad, netD2.parameters())),lr=0.0001,betas=(0.6, 0.95),amsgrad=True)
CE_loss = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss()
lossD_all=0
lossG_all=0
batch_size=10

for epoch in range(10):
	for i in tqdm.trange(5001):
		z = torch.randn(batch_size, in_dim).to(device)
#Training D
		with torch.no_grad():
			x = netG2(z,depth=8,alpha=1)
		z_ = netD2(x.detach(),height=8,alpha=1)
		z_ = z_.squeeze(2).squeeze(2)
		x_ = netG2(z_,depth=8,alpha=1)#通过G训练D
		optimizerD.zero_grad()
		loss_i = MSE_loss(x_,x)
		loss_i.backward()
		optimizerD.step()
		lossD_all +=loss_i.item()
		with open(resultPath+'/LossD.txt', 'a+') as f:
			print('loss_all__:  '+str(lossD_all)+'     loss_i:    '+str(loss_i.item()),file=f)

#Training G:
		z_dim = np.random.randint(in_dim)
		z_2 = torch.zeros(batch_size, in_dim).to(device)
		z_2[:,z_dim] = 1
		x_2 = netG2(z_2,depth=8,alpha=1)
		z_d = netD2(x_2.detach(),height=8,alpha=1)#通过D训练G
		z_d = z_d.squeeze(2).squeeze(2)
		optimizerG.zero_grad()
		loss_j = CE_loss(z_d, torch.tensor(np.repeat(z_dim*1.0,batch_size)).long().to(device))
		loss_j.backward()
		optimizerG.step()
		lossG_all +=loss_j.item()
		with open(resultPath+'/LossG.txt', 'a+') as f:
			print('loss_all__:  '+str(lossG_all)+'     loss_j:    '+str(loss_j.item()),file=f)
#Inference
		if i % 100 == 0: 
			#测试重构
			img = (torch.cat((x[:8],x_[:8]))+1)/2
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d_rc.jpg'%(epoch,i), nrow=8)
			#测试解耦
			temp = torch.linspace(-4,4,8)
			z_2_2=z_2
			if z_dim+1 != in_dim:
				z_2_2[:8,z_dim+1] = temp
			else:
				z_2_2[:8,0] = temp
			z_2[:8,z_dim] = temp
			with torch.no_grad():
				img2_1 = netG2(z_2[:8],depth=8,alpha=1) 
				img2_2 = netG2(z_2_2[:8],depth=8,alpha=1)
			img2 = (torch.cat((img2_1,img2_2))+1)/2
			torchvision.utils.save_image(img2, resultPath1_1+'/ep%d_%d_dim.jpg'%(epoch,i), nrow=8)
	#if epoch%10==0 or epoch == 29:
	torch.save(netG2.state_dict(), resultPath1_2+'/G_model_ep%d.pth'%epoch)
	torch.save(netD2.state_dict(), resultPath1_2+'/D_model_ep%d.pth'%epoch)
	print('finish-ep%d'%epoch)


#改进: D和E 分开训练
# D复杂产生更好的编码，帮助G解码生成更好的图像
# E负责更好的解码，帮助D更好的完成编码解藕s

















