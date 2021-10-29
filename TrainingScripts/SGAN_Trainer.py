import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import HalfTensor, LongTensor
import sys
sys.path.append("../")
from Models.Generator import Generator
from Models.ResNet import ResnetClassifier, ResnetConfig
from Data.Data import DiffractionDataset

class Saver:
    def __init__(self):
        now=datetime.now()
        self.dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        self.path=r"../Generated_imgs/"+self.dt_string
        os.mkdir(self.path)
    def save(self,epoch, img):
        torch.save(img,self.path+"/{}.pt".format(epoch))
    def get_time(self):
        return self.dt_string

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
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '7013'
    dist.init_process_group(backend, rank=rank, world_size=size)

def init_models(gpu):
    generator=Generator(32, 32)
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
    generator=generator.half()
    discriminator=discriminator.half()
    generator.cuda(gpu)
    discriminator.cuda(gpu)
    
    opt_G=torch.optim.Adam(generator.parameters(), lr=1e-4, eps=1e-4)
    opt_SD=torch.optim.Adam(discriminator.parameters(), lr=1e-3, eps=1e-4)
    opt_UD=torch.optim.Adam(discriminator.parameters(), lr=1e-4, eps=1e-4)
    return generator, discriminator, opt_G, opt_SD, opt_UD

def train(gpu, epochs, world_size, batch_size, dset, data_dim=3041):
    setup_start = datetime.now()
    rank=gpu
    init_process(gpu, world_size)
    print(
        f"Rank {gpu + 1}/{world_size} process initialized.\n"
    )
    torch.manual_seed(0)
    
    generator, discriminator, optimizer_G, optimizer_SD, optimizer_UD=init_models(gpu)
    
    torch.cuda.set_device(gpu)
    torch.autograd.set_detect_anomaly(True)
    
    generator = DDP(generator, device_ids=[gpu])
    discriminator = DDP(discriminator, device_ids=[gpu], find_unused_parameters = True)
    supervised_dataset=torch.load("path to supervised data")
    unsup_dataset=torch.load("path to unsupervised data", unsupervised=True)
    if gpu==0:
        test_dataset=DiffractionDataset("path to testing data")
        
    supervised_dataset=DiffractionDataset(supervised_dataset["X"]+1e-3, supervised_dataset["Y"], False)
    unsup_dataset=DiffractionDataset(unsup_dataset["X"]+1e-3)
    
    train_sampler=torch.utils.data.distributed.DistributedSampler(supervised_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    
    train_loader = torch.utils.data.DataLoader(dataset=supervised_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    adv_loss=torch.nn.BCEWithLogitsLoss().cuda(gpu)
    aux_loss=torch.nn.CrossEntropyLoss().cuda(gpu)
    start = datetime.now()
    v_acc=0
    clip_val=1
    if gpu==0:
        print("Setup time: " + str(datetime.now() - setup_start))
        print("Generator parameters: {}".format(sum(p.numel() for p in generator.parameters() if p.requires_grad)))
        print("Discriminator parameters: {}".format(sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))
        s=Saver()
        L=Logger(s.get_time())
    z = torch.normal(0, 1, (batch_size, 1, 100)).cuda(gpu).half()
    for epoch in range(epochs):
        valid = 0.2*torch.rand((batch_size, 1, 1)).half().cuda(gpu)
        fake = 0.8+0.2*torch.rand((batch_size, 1, 1)).half().cuda(gpu)

        if gpu==0:
            discriminator.train()
        for i, (imgs, labels) in enumerate(train_loader):
            real_imgs = torch.reshape(imgs, (imgs.shape[0], 1, data_dim)).cuda(gpu)
            labels = labels.type(LongTensor).cuda(gpu)
            unsup_imgs=torch.reshape(unsup_dataset.batch_u(batch_size), (batch_size, 1, data_dim)).cuda(gpu)
            optimizer_G.zero_grad()
            z = torch.normal(0, 1, (batch_size, 1, 100)).cuda(gpu).half()
            gen_imgs = generator(z).logits.half()
            g = discriminator(gen_imgs, labels=valid, s=False, loss_func=adv_loss)
            g_loss=g.loss
            g_loss.backward()
            optimizer_G.step()
            
            gen_imgs=gen_imgs.detach()

            optimizer_SD.zero_grad()
            r_s = discriminator(real_imgs, labels=labels, s=True, loss_func=aux_loss)
            r_s.loss.backward()
            optimizer_SD.step()

            
            optimizer_UD.zero_grad()
            r_u = discriminator(unsup_imgs, labels = valid, s=False, loss_func=adv_loss)
            g_u = discriminator(gen_imgs, labels=fake, s=False, loss_func=adv_loss)
            u_loss=g_u.loss+r_u.loss
            u_loss.backward()
            optimizer_UD.step()
            
            d_acc=torch.mean((torch.flatten(torch.argmax(r_s.logits, axis=-1))==labels).float())
            
            if gpu==0:
                plot_imgs=gen_imgs[0]
                print("[Epoch %d/%d] [%d] [SD loss: %.2f  acc: %d%%] [UD L: %.2f] [G L: %.2f]"
                % (epoch+1, epochs,i, r_s.loss.item(), 100 * d_acc, u_loss.item(),g.loss.item()))
            del real_imgs, labels, unsup_imgs, gen_imgs, r_u, g_u, g
            if gpu!=0:
                del d_acc, u_loss
            torch.cuda.empty_cache()
        if gpu==0:
            s.save(epoch, plot_imgs)
            discriminator.eval()
            t_data=torch.reshape(test_dataset.data, (test_dataset.data.shape[0], 1,data_dim))
            pred=discriminator(t_data, labels=test_dataset.labels.long().cuda(gpu), s=True, loss_func=aux_loss)
            v_acc=torch.mean((torch.flatten(torch.argmax(pred.logits, axis=-1))==test_dataset.labels.cuda(gpu)).half())
            L.log(
                "[Epoch %d/%d] [SD loss: %.2f  acc: %d%%] [UD W: %.2f] [G W: %.2f] [T loss: %.2f acc: %d%%]\n"
                % (epoch+1, epochs, r_s.loss.item(), 100 * d_acc, u_loss.item(),
                   g_loss.item(),pred.loss.item(),100*v_acc)
            )
            del d_acc, u_loss, g_loss, pred, t_data
            torch.cuda.empty_cache()
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        torch.save({
            'generator': generator.state_dict(),
            'opt_G': optimizer_G.state_dict(),
            'discriminator': discriminator.state_dict(),
            'opt_SD': optimizer_SD.state_dict(),
            'opt_UD': optimizer_UD.state_dict(),
            'acc': v_acc
            }, r"SavedModels/sg_sgan_net_"+dset+".pt")
        
def spawn(epochs, world_size,batch_size,dset='50%', train=train):
    mp.spawn(
            train, args=(epochs, world_size, batch_size, dset),
            nprocs=world_size, join=True
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('-e','--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    world_size = args.gpus * args.nodes
    epochs = args.epochs
    spawn(epochs, world_size, args.batch_size)

if __name__ == '__main__':
    main()
