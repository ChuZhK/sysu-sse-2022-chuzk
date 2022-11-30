from config import *
from model import *
from dataset import *

import ir152 as ir152

import sys
import os
import torchvision
sys.path.append('./')
from mytraindata import *
from mydataset import *

config = Configuration()
device = torch.device('cuda:{}'.format(config.gpu)) if config.gpu >= 0 else torch.device('cpu')
enc=Encoder(3).to(device)


def my_train(model,train_loader):
    
        for it, data in enumerate(train_loader):
            print("\n\nin my_train load data.....\n",data.to(device).size())
            t1=data.to(device).detach()
            print("\n\nt1 -----> t1\n",t1.size())
            print(t1)
            #before = enc(t1, bn_training=False)
            #print(data[0].to(device).detach())
            #print(before[0].size())
            model(t1)
            
               
        

print(torch.cuda.is_available())
#model=torch.load('./ir152.pth',map_location=device)
#print("\n\nir152 is size: ......\n",ir152.IR_152((112,112)))
dataset = dataset_makeup(config)
train_loader =torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=config.n_threads)
fr_model = ir152.IR_152((112,112))
fr_model.to(device)
fr_model.eval()

criterion = torch.nn.MSELoss(reduction='sum') 
optimizer = torch.optim.SGD(fr_model.parameters(), lr=1e-4)

my_train(fr_model,train_loader)