import os
from datetime import datetime
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.path.append("../")
from Models.ResNet import ResnetClassifier, ResnetConfig
from Data.Data import DiffractionDataset

class Logger:
    def __init__(self, time):
        self.time=time
        self.path=r"../OutputFiles/" + self.time+".txt"
    def log(self, msg):
        f=open(self.path, 'a')
        f.write(msg)
        f.close()
    def get_path(self):
        return r"../OutputFiles/" + self.time

def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '7017'
    dist.init_process_group(backend, rank=rank, world_size=size)

def init_models(gpu):
    config = ResnetConfig(
        input_dim = 1,
        output_dim = 144,
        res_dims=[32, 64, 64, 64],
        res_kernel=[5, 7, 17, 13],
        res_stride=[4, 4, 5, 3],
        num_blocks=[2, 2, 2, 2],
        first_kernel_size = 13,
        first_stride = 1,
        first_pool_kernel_size = 7,
        first_pool_stride = 7,
    )
    discriminator = ResnetClassifier(config)
    opt_SD=torch.optim.Adam(discriminator.parameters(), lr=1e-3, eps=1e-4)
    discriminator=discriminator.half()    
    return discriminator, opt_SD
def train(gpu, epochs, world_size, batch_size):
    setup_start = datetime.now()
    rank=gpu
    init_process(gpu, world_size)
    print(
        f"Rank {gpu + 1}/{world_size} process initialized.\n"
    )
    torch.manual_seed(0)
    discriminator, optimizer_SD = init_models(gpu)
    torch.cuda.set_device(gpu)
    discriminator.cuda(gpu)
    torch.autograd.set_detect_anomaly(True)
    discriminator = DDP(discriminator, device_ids=[gpu], find_unused_parameters=True)
    supervised_dataset=torch.load("path to training data")
    if gpu==0:
        test_dataset=torch.load("path to testing data")
        test_dataset=DiffractionDataset(test_dataset["X"]+1e-3, test_dataset["Y"], False)
    supervised_dataset=DiffractionDataset(supervised_dataset['X']+1e-3, supervised_dataset['Y'], False)

    train_sampler=torch.utils.data.distributed.DistributedSampler(supervised_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    
    train_loader = torch.utils.data.DataLoader(dataset=supervised_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    aux_loss=torch.nn.CrossEntropyLoss().cuda(gpu)
    start = datetime.now()
    v_acc=0
    if gpu==0:
        print("Setup time: " + str(datetime.now() - setup_start))
        print("Discriminator parameters: {}".format(sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))
        L=Logger(str(start))
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            if gpu==0:
                discriminator.train()
            imgs = imgs.cuda(gpu)
            labels = labels.long().cuda(gpu)
            optimizer_SD.zero_grad()
            r_s = discriminator(imgs, labels=labels, s=True, loss_func=aux_loss)
            s_loss=r_s.loss
            s_loss.backward()
            optimizer_SD.step()
            d_acc=torch.mean((torch.flatten(torch.argmax(r_s.logits, axis=-1))==labels).float())
            
            if gpu==0:
                print("[Epoch %d/%d] [%d] [SD loss: %.2f  acc: %d%%]"
                % (epoch+1, epochs,i, s_loss.item(), 100 * d_acc))
        if gpu==0:
            discriminator.eval()
            t_data=test_dataset.data.cuda(gpu)
            pred=discriminator(t_data, labels=test_dataset.labels.long().cuda(gpu), s=True, loss_func=aux_loss)
            v_acc=torch.mean((torch.flatten(torch.argmax(pred.logits, axis=-1))==test_dataset.labels.cuda(gpu)).half())
            L.log(
                "[Epoch %d/%d] [SD loss: %.2f  acc: %d%%] [T loss: %.2f acc: %d%%]\n"
                % (epoch+1, epochs, s_loss.item(), 100 * d_acc, pred.loss.item(),100*v_acc)
            )
            del s_loss, d_acc, pred, t_data
            torch.cuda.empty_cache()
            torch.save({'model': discriminator.state_dict(),
            'optimizer': optimizer_SD.state_dict()
            }, 'supervised_net.pt')
def spawn(epochs, world_size,batch_size, train=train):
    mp.spawn(
            train, args=(epochs, world_size, batch_size),
            nprocs=world_size, join=True
        )