import time
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from DistOptmizers import CommOptimizer
from DistOptmizers import to_dist_train_loader, sync_model_params_and_buffers

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(  1, 16, 3, 2), nn.ReLU(),
            nn.Conv2d( 16, 32, 3, 2), nn.ReLU(),
            nn.Conv2d( 32, 64, 3, 2), nn.ReLU(),
        )   
        self.classifier = nn.Sequential(
            nn.Linear( 256, 100), nn.ReLU(),
            nn.Linear( 100,  10)
        )

    def forward(self, x):
        x = self.base(x)
        output = self.classifier(torch.flatten(x, 1))
        return output

def train(model, device, train_loader, optimizer, scheduler, epoch):
    train_begin_time = time.time()
    model.train()
    try:
        train_loader.sampler.set_epoch(epoch)
    except:
        pass
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
    return time.time() - train_begin_time

def test(model, device, test_loader):
    center_model = copy.deepcopy(model)
    sync_model_params_and_buffers(center_model)
    test_loss = test_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = center_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy

def dist_run(rank, accm_workers, world_size, init_method):
    torch.manual_seed(0)
    dist.init_process_group(rank=accm_workers+rank, world_size=world_size, backend='nccl', init_method=init_method)

    cuda_kwargs  = {'num_workers': 4}
    train_kwargs = {'batch_size': 128}
    test_kwargs  = {'batch_size': 1000}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_download = dist.get_rank() % torch.cuda.device_count() == 0
    Dataset = datasets.FashionMNIST
    if dataset_download:
        dataset1 = Dataset('./data', train=True, download=dataset_download)
        dataset2 = Dataset('./data', train=False, download=dataset_download)
    dist.barrier()
    dataset1 = Dataset('./data', train=True, download=False, transform=transform)
    dataset2 = Dataset('./data', train=False, download=False, transform=transform)
    train_loader = to_dist_train_loader(torch.utils.data.DataLoader(dataset1, **train_kwargs), is_dist_sampler=True)
    test_loader  = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    device = torch.device('cuda:%d'%rank)
    base_model = NeuralNetwork().to(device)
    for model_param in base_model.parameters():
        dist.all_reduce_multigpu([model_param.data], op=torch.distributed.ReduceOp.AVG, async_op=False)

    for dist_optim_name in ('DataParallel', 'LSGD', 'LSGD+', 'EASGD', 'EASGD+', ):
        # define optimizer
        if not dist_optim_name == 'DataParallel':
            model     = copy.deepcopy(base_model).to(device)
            optimizer = optim.SGD(model.parameters(), lr = 0.2, momentum = 0.9, weight_decay=1e-5)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
            optimizer = CommOptimizer(optimizer, dist_optim_name=dist_optim_name, comm_period=4, dist_pulling_strength=0.1, local_pulling_strength=0.1)
        else:
            model     = nn.parallel.DistributedDataParallel(copy.deepcopy(base_model).to(device))
            optimizer = optim.SGD(model.parameters(), lr = 0.2, momentum = 0.9, weight_decay=1e-5)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        
        # train model
        total_seconds = [0]; all_test_loss = []; all_test_accuracy = []
        for epoch in range(1, 20 + 1):
            train_seconds = train(model, device, train_loader, optimizer, scheduler, epoch)
            total_seconds += [train_seconds + total_seconds[-1]]
            test_loss, test_accuracy = test(model, device, test_loader)
            all_test_loss += [test_loss]; all_test_accuracy += [test_accuracy]
            if dist.get_rank() % torch.cuda.device_count() == 0:
                print('%s (epoch: %d | seconds: %.2f) te_loss: %.3f te_acc: %.3f%%'\
                    %(dist_optim_name, epoch, total_seconds[-1], test_loss, 100. * test_accuracy))

        # plot resutls
        plt.plot(range(1, len(all_test_accuracy) + 1), [acc * 100. for acc in all_test_accuracy], label=dist_optim_name, marker='.',linewidth=2)
        plt.xticks(range(1, len(all_test_accuracy) + 1)); plt.legend(); plt.grid(True); 
        plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.title('Test Accuracy')
        plt.savefig('test_results.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_address', type=str, required=True)
    parser.add_argument('--node_rank',  type=int, default = 0) 
    parser.add_argument('--node_gpus',  type=int, default = 100) 
    parser.add_argument('--num_nodes', type=int, default = 1)
    args = parser.parse_args()

    cur_workers   = min(args.node_gpus, torch.cuda.device_count())
    total_workers = torch.zeros(args.num_nodes).int()
    total_workers[args.node_rank] = cur_workers
    if args.num_nodes > 1:
        init_method = "tcp://{ip}:{port}0".format(ip=args.ip_address, port=2432)
        dist.init_process_group(rank=args.node_rank, world_size=args.num_nodes, backend='gloo', init_method=init_method)
        dist.all_redcude(total_workers, op=torch.distributed.ReduceOp.SUM, async_op=False)
    accm_workers = total_workers[:args.node_rank].sum().item()
    world_size   = total_workers.sum().item()

    init_method = "tcp://{ip}:{port}5".format(ip=args.ip_address, port=2432)
    mp.spawn(fn=dist_run, args=(accm_workers, world_size, init_method), nprocs=cur_workers)