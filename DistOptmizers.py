import torch
import torch.distributed as dist
import torch.utils.data as udata

def sync_model_params_and_buffers(model):
    ''' synchronize model params and buffers '''
    current_index = 0
    for parameter in model.parameters():
        numel = parameter.data.numel()
        dist.all_reduce_multigpu([parameter.data], op=torch.distributed.ReduceOp.AVG, async_op=False)
        current_index += numel

    current_index = 0
    for parameter in model.buffers():
        numel = parameter.data.numel()
        dist.all_reduce_multigpu([parameter.data], op=torch.distributed.ReduceOp.AVG, async_op=False)
        current_index += numel

def to_dist_train_loader(train_loader, is_dist_sampler=False):
    ''' convert local train data loader to distributed train data loader by modifying the arguments '''
    if is_dist_sampler:
        train_sampler = udata.distributed.DistributedSampler(train_loader.dataset, shuffle=True, drop_last=train_loader.drop_last)
        return udata.DataLoader(dataset=train_loader.dataset,
                                batch_size=train_loader.batch_size,
                                num_workers=train_loader.num_workers,
                                pin_memory=True, persistent_workers=True, sampler = train_sampler)
    else:
        return udata.DataLoader(dataset=train_loader.dataset,
                                drop_last=train_loader.drop_last,
                                batch_size=train_loader.batch_size,
                                num_workers=train_loader.num_workers,
                                pin_memory=True, persistent_workers=True, shuffle=True)

class CommOptimizer:
    def __init__(self, local_optimizer, dist_optim_name, local_pulling_strength, dist_pulling_strength, comm_period):
        ''' combine both local training and distributed training optimziers '''
        self.local_optimizer = local_optimizer
        if dist_optim_name == 'LSGD':
            self.dist_optimizer = LSGD(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)
        if dist_optim_name == 'LSGD+':
            self.dist_optimizer = LSGDPlus(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)
        elif dist_optim_name == 'EASGD':
            self.dist_optimizer = EASGD(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)
        elif dist_optim_name == 'EASGD+':
            self.dist_optimizer = EASGDPlus(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)

    def zero_grad(self):
        self.local_optimizer.zero_grad()
        
    def step(self, closure=None):
        self.local_optimizer.step(closure)
        if not self.dist_optimizer is None:
            self.dist_optimizer.step()

class DistOptimizer:
    def __init__(self, local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period):
        self.dist_counter = 0
        self.comm_period = comm_period
        self.local_pulling_strength = local_pulling_strength
        self.dist_pulling_strength = dist_pulling_strength
        self.local_optimizer = local_optimizer
        
        self.param_tensor_list = []
        self.center_tensor_list = []
        self.device_list = []
        for param_group in local_optimizer.param_groups:
            param_list = list(param_group['params'])
            num_params = sum([p.numel() for p in param_list])
            device = param_list[0].device; self.device_list += [device]
            param_tensor = torch.zeros(num_params, device=device)
            counter = 0
            for model_param in param_list:
                param_tensor[counter:counter+model_param.numel()].copy_(model_param.view(-1))
                counter += model_param.numel()
            self.param_tensor_list += [param_tensor]
            self.center_tensor_list += [param_tensor.clone()]
            
    def local_pulling_step(self):
        p = self.local_pulling_strength / self.comm_period
        for param_group, center_tensor in zip(self.local_optimizer.param_groups, self.center_tensor_list):
            counter = 0
            for param in param_group['params']:
                param.data.mul_(1-p).add_(center_tensor[counter:counter+param.numel()].view_as(param), alpha=p)
                counter = counter + param.numel()

    def step(self):
        self.dist_counter += 1
        self.local_pulling_step()
        self.dist_train_step()
        if self.dist_counter % self.comm_period == 0: # dist training
            self.comm_train_step()
            self.dist_counter = 0
    
    def comm_train_step(self):
        raise NotImplementedError

    def dist_train_step(self): 
        raise NotImplementedError
    
class LSGD(DistOptimizer):
    def __init__(self, local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period):
        super().__init__(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)
        self.dist_message_tensor_list = []
        self.dist_loss_tensor_list = []
        for device in self.device_list:
            self.dist_message_tensor_list += [torch.zeros(torch.distributed.get_world_size(), device=device)]
            self.dist_loss_tensor_list += [torch.zeros(1, device=device)]

    def dist_train_step(self):
        # copy model parameters into dist tensor
        for param_group, param_tensor, loss in zip(self.local_optimizer.param_groups, self.param_tensor_list, self.dist_loss_tensor_list):
            counter = 0; group_loss = 0
            for param in param_group['params']:
                with torch.no_grad():
                    cur_step = param_tensor[counter:counter+param.data.numel()] - param.data.view(-1)
                    group_loss += (cur_step * param.grad.view(-1)).sum().item()
                param_tensor[counter:counter+param.numel()].copy_(param.view(-1))
                counter += param.numel()
            loss.mul_(0.9).add_(group_loss, alpha=0.1)

    def comm_train_step(self):
        for param_group, param_tensor, center_tensor, message, loss in \
            zip(self.local_optimizer.param_groups, self.param_tensor_list, self.center_tensor_list, \
                self.dist_message_tensor_list, self.dist_loss_tensor_list):
                    # find the rank of leader
                    message.zero_(); message[dist.get_rank()] = loss.item()
                    dist.all_reduce_multigpu([message], op=torch.distributed.ReduceOp.SUM, async_op=False)
                    leader_rank = message.argmin().item()
                    dist.broadcast_multigpu([param_tensor], src=leader_rank, async_op=False)

                    # update model parameters
                    counter = 0
                    for param in param_group['params']:    
                        param.data.mul_(1 - self.dist_pulling_strength)
                        param.data.add_(param_tensor[counter:counter+param.numel()].view_as(param), alpha=self.dist_pulling_strength)
                        param_tensor[counter:counter+param.numel()].copy_(param.data.view(-1))
                        counter += param.numel()
                    center_tensor.copy_(param_tensor)

class LSGDPlus(LSGD):
    def __init__(self, local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period):
        super().__init__(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)

    def comm_train_step(self):
        for param_group, param_tensor, center_tensor, message, loss in \
            zip(self.local_optimizer.param_groups, self.param_tensor_list, self.center_tensor_list, \
                self.dist_message_tensor_list, self.dist_loss_tensor_list):
                    # find the average of workers
                    center_tensor.copy_(param_tensor)
                    dist.all_reduce_multigpu([center_tensor], op=torch.distributed.ReduceOp.AVG, async_op=False)

                    # find the rank of leader
                    message.zero_(); message[dist.get_rank()] = loss.item()
                    dist.all_reduce_multigpu([message], op=torch.distributed.ReduceOp.SUM, async_op=False)
                    leader_rank = message.argmin().item()
                    # if dist.get_rank() == leader_rank:
                    #     m_weight = 1 + self.dist_pulling_strength * (dist.get_world_size() - 1)
                    # else:
                    #     m_weight = 1 - self.dist_pulling_strength
                    # param_tensor = m_weight * param_tensor
                    # dist.all_reduce_multigpu([param_tensor], op=torch.distributed.ReduceOp.AVG, async_op=False)
                    dist.broadcast_multigpu([param_tensor], src=leader_rank, async_op=False)
                    delta_tensor = self.dist_pulling_strength * (param_tensor - center_tensor)
                    center_tensor.add_(delta_tensor)

                    # update model parameters
                    counter = 0
                    for param in param_group['params']:    
                        param.data.add_(delta_tensor[counter:counter+param.numel()].view_as(param))
                        param.data.mul_(1-self.dist_pulling_strength)
                        param.data.add_(center_tensor[counter:counter+param.numel()].view_as(param), alpha=self.dist_pulling_strength)
                        param_tensor[counter:counter+param.numel()].copy_(param.data.view(-1))
                        counter += param.numel()

class EASGD(DistOptimizer):
    def __init__(self, local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period):
        super().__init__(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)
        self.dist_message_tensor_list = []
        self.dist_loss_tensor_list = []

    def dist_train_step(self):
        pass

    def comm_train_step(self):
        for param_group, param_tensor, center_tensor in zip(self.local_optimizer.param_groups, self.param_tensor_list, self.center_tensor_list):
            # find average of model parameters
            counter = 0
            for param in param_group['params']:
                param_tensor[counter:counter+param.numel()].copy_(param.view(-1))
                counter += param.numel()
            dist.all_reduce_multigpu([param_tensor], op=torch.distributed.ReduceOp.AVG, async_op=False)

            # update model parameters
            counter = 0
            for param in param_group['params']:
                param.data.mul_(1-self.dist_pulling_strength)
                param.data.add_(center_tensor[counter:counter+param.numel()].view_as(param), alpha=self.dist_pulling_strength)
                counter += param.numel()
            center_tensor.copy_(param_tensor)

class EASGDPlus(EASGD):
    def __init__(self, local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period):
        super().__init__(local_optimizer, local_pulling_strength, dist_pulling_strength, comm_period)
        self.grad_tensor_list = []
        for param_tensor in self.param_tensor_list:
            self.grad_tensor_list += [torch.zeros_like(param_tensor)]
            
    def dist_train_step(self):
        for param_group, grad_tensor in zip(self.local_optimizer.param_groups, self.grad_tensor_list):
            counter = 0
            for param in param_group['params']:
                grad_tensor[counter:counter+param.numel()].mul_(0.9).add_(param.grad.square().view(-1), alpha=0.1)
                counter += param.numel()

    def comm_train_step(self):
        for param_group, param_tensor, grad_tensor, center_tensor in \
            zip(self.local_optimizer.param_groups, self.param_tensor_list, self.grad_tensor_list, self.center_tensor_list):
                # find the average of workers
                center_tensor.copy_(param_tensor)
                dist.all_reduce_multigpu([center_tensor], op=torch.distributed.ReduceOp.AVG, async_op=False)

                # find weighted average of model parameters
                counter = 0
                for param in param_group['params']:
                    param_tensor[counter:counter+param.numel()].copy_(param.view(-1))
                    counter += param.numel()
                dist_grad_tensor = grad_tensor + 1e-8
                param_tensor.mul_(dist_grad_tensor)
                dist.all_reduce_multigpu([param_tensor], op=torch.distributed.ReduceOp.SUM, async_op=False)
                dist.all_reduce_multigpu([dist_grad_tensor], op=torch.distributed.ReduceOp.SUM, async_op=False)
                param_tensor.div_(dist_grad_tensor)
                delta_tensor = self.dist_pulling_strength * (param_tensor - center_tensor)
                center_tensor.add_(delta_tensor)

                # update model parameters
                counter = 0
                for param in param_group['params']:
                    param.data.add_(delta_tensor[counter:counter+param.numel()].view_as(param))
                    param.data.mul_(1-self.dist_pulling_strength)
                    param.data.add_(center_tensor[counter:counter+param.numel()].view_as(param), alpha=self.dist_pulling_strength)
                    param_tensor[counter:counter+param.numel()].copy_(param.data.view(-1))
                    counter += param.numel()