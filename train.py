import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from trainer import Trainer, Trainer_DDP
from data.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image, random_ff_mask
from utils.logger import get_logger
from utils.tools import get_model_list, local_patch, spatial_discounting_mask
from torch import autograd
from model.networks import Generator, LocalDis, GlobalDis, PatchDis


from tqdm.auto import tqdm
from datetime import datetime


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--load_ckpt', type=str, default=None)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--distributed", action='store_true')



def gan_hinge_loss(pos, neg, value=1.):
    '''
    take from official implementation by author,
    https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/gan_ops.py
    
    Args:
        pos: The real, or true outputs examples.
        neg: The fake, or generated outputs.
        value: boundary max.
    
    Returns:
        d_loss: Discriminator loss
        g_loss: Generator loss
    '''

    relu = nn.ReLU()
    hinge_pos = torch.mean(relu(1 - pos))
    hinge_neg = torch.mean(relu(1 + neg))
    d_loss = torch.add(0.5 * hinge_pos, 0.5 * hinge_neg)
    g_loss = -1.0 * neg.mean()
    
    return g_loss, d_loss


def save_model(netG, globalD, localD, optG, optD, checkpoint_dir, iteration):
    # Save generators, discriminators, and optimizers
    #
    ckpt_name = os.path.join(checkpoint_dir, 'ckpt_%08d.pt' % iteration)
    torch.save(
        {'netG': netG.state_dict(),
        'globalD': globalD.state_dict(),
        'localD': localD.state_dict(),
        'optG': optG.state_dict(),
        'optD': optD.state_dict()},
        ckpt_name
    )
    
    
def save_model_v2(netG, patchD, optG, optD, checkpoint_dir, iteration):
    # Save generators, discriminators, and optimizers
    #
    ckpt_name = os.path.join(checkpoint_dir, 'ckpt_%08d.pt' % iteration)
    torch.save(
        {'netG': netG.state_dict(),
        'patchD': patchD.state_dict(),
        'optG': optG.state_dict(),
        'optD': optD.state_dict()},
        ckpt_name
    )
    
def calc_gradient_penalty(netD, real_data, fake_data, local_rank):
    # Calculate gradient penalty
    #
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data).cuda(local_rank)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_().clone()

    disc_interpolates = netD(interpolates)
    grad_outputs = torch.ones(disc_interpolates.size()).cuda(local_rank)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def discriminator_pred(netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred


def forward(config, x, bboxes, masks, ground_truth,
            netG, localD, globalD,
            local_rank, compute_loss_g=False):

        l1_loss = nn.L1Loss().cuda(local_rank)
        losses = {}


        x1, x2, offset_flow = netG(x, masks)

        
        local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)
        
        
#         # D part
#         # wgan d loss
#         local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
#             self.localD, local_patch_gt, local_patch_x2_inpaint.detach()
#         )
        batch_size = local_patch_gt.size(0)
        batch_data = torch.cat([local_patch_gt, local_patch_x2_inpaint.detach()], dim=0)
        batch_output = localD(batch_data)
        local_patch_real_pred, local_patch_fake_pred = torch.split(batch_output, batch_size, dim=0)
        
        
#         global_real_pred, global_fake_pred = self.dis_forward(
#             self.globalD, ground_truth, x2_inpaint.detach()
#         )
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x2_inpaint.detach()], dim=0)
        batch_output = globalD(batch_data)
        global_real_pred, global_fake_pred = torch.split(batch_output, batch_size, dim=0)
        
        
        losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
            torch.mean(global_fake_pred - global_real_pred) * config['global_wgan_loss_alpha']
        
        
        # gradient penalty loss
        #
        local_penalty = calc_gradient_penalty(
            localD, local_patch_gt, local_patch_x2_inpaint.detach(), local_rank
        )
        
        global_penalty = calc_gradient_penalty(
            globalD, ground_truth, x2_inpaint.detach(), local_rank
        )
        
        losses['wgan_gp'] = local_penalty + global_penalty
        
        
        # G part
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(config)
            losses['l1'] = l1_loss(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) * \
                config['coarse_l1_alpha'] + \
                l1_loss(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
            
            losses['ae'] = l1_loss(x1 * (1. - masks), ground_truth * (1. - masks)) * \
                config['coarse_l1_alpha'] + \
                l1_loss(x2 * (1. - masks), ground_truth * (1. - masks))

            # wgan g loss
            local_patch_real_pred, local_patch_fake_pred = discriminator_pred(
                localD, local_patch_gt, local_patch_x2_inpaint
            )
            
            global_real_pred, global_fake_pred = discriminator_pred(
                globalD, ground_truth, x2_inpaint
            )
            
            losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                torch.mean(global_fake_pred) * config['global_wgan_loss_alpha']

        
        return losses, x2_inpaint, offset_flow
        
        
        
def train_distributed_v2(config, logger, writer, checkpoint_path):
    
    dist.init_process_group(                                   
        backend='nccl',
#         backend='gloo',
        init_method='env://'
    )  
    
    
    # Find out what GPU on this compute node.
    #
    local_rank = torch.distributed.get_rank()
    
    
    # this is the total # of GPUs across all nodes
    # if using 2 nodes with 4 GPUs each, world size is 8
    #
    world_size = torch.distributed.get_world_size()
    print("### global rank of curr node: {} of {}".format(local_rank, world_size))
    
    
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    #
    print("local_rank: ", local_rank)
#     dist.barrier()
    torch.cuda.set_device(local_rank)
    print("Creating models on device: ", local_rank)
    
    
    # Various definitions for models, etc.
    #
    input_dim = config['netG']['input_dim']
    cnum = config['netG']['ngf']
    use_cuda = True
    gated = config['netG']['gated']
    batch_size = config['batch_size']
    
    
    # L1 loss used on outputs from course and fine networks in generator.
    #
    loss_l1 = nn.L1Loss(reduction='mean').cuda()
    
    # Models
    #
    netG = Generator(config['netG'], use_cuda=True, device=local_rank).cuda()
    netG = torch.nn.parallel.DistributedDataParallel(
        netG,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    
    patchD = PatchDis(config['netD'], use_cuda=True, device=local_rank).cuda()
    patchD = torch.nn.parallel.DistributedDataParallel(
        patchD,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    
    
    if local_rank == 0:
        logger.info("\n{}".format(netG))
        logger.info("\n{}".format(patchD))
        
    
    # Optimizers
    #
    optimizer_g = torch.optim.Adam(
        netG.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2'])
    )
    
    optimizer_d = torch.optim.Adam(
        patchD.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2'])
    )
    
    if local_rank == 0:
        logger.info("\n{}".format(netG))
        logger.info("\n{}".format(patchD))
    
    
    # Data
    #
    sampler = None
    train_dataset = Dataset(
        data_path=config['train_data_path'],
        with_subfolder=config['data_with_subfolder'],
        image_shape=config['image_shape'],
        random_crop=config['random_crop']
    )
        
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
#             num_replicas=torch.cuda.device_count(),
        num_replicas=len(config['gpu_ids']),
#         rank = local_rank
    )
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=config['num_workers'],
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    
    
    losses = {
        'coarse': 0.0,
        'fine': 0.0,
        'ae': 0.0,
        'g_loss': 0.0,
        'd_loss': 0.0
    }
    
    # Get the resume iteration to restart training
    #
    ### TODO:
    ### - allow resuming from checkpoint.
    ###
#     start_iteration = trainer.resume(config['resume']) if config['resume'] else 1
    start_iteration = 1
    print("\n\nStarting epoch: ", start_iteration)

    iterable_train_loader = iter(train_loader)

    if local_rank == 0: 
        time_count = time.time()

    epochs = config['niter'] + 1
    pbar = tqdm(range(start_iteration, epochs), dynamic_ncols=True, smoothing=0.01)
    for iteration in pbar:
        sampler.set_epoch(iteration)
        
        try:
            ground_truth = next(iterable_train_loader)
        except StopIteration:
            iterable_train_loader = iter(train_loader)
            ground_truth = next(iterable_train_loader)
    
        ground_truth = ground_truth.cuda(local_rank)
        mask_ff = random_ff_mask(config['random_ff_settings'], batch_size=batch_size).cuda(local_rank)
        
#         netG.zero_grad()
        imgs_incomplete = ground_truth * (1. - mask_ff) # just background
        x1, x2, offset_flow = netG(imgs_incomplete, mask_ff)
        imgs_complete = (x2 * mask_ff) + imgs_incomplete
        
        
        # Losses 
        #
        coarse_loss = config['l1_loss_alpha'] * loss_l1(ground_truth, x1)
        fine_loss = config['l1_loss_alpha'] * loss_l1(ground_truth, x2)
        ae_loss = coarse_loss + fine_loss
        losses['coarse'] = coarse_loss.item()
        losses['fine'] = fine_loss.item()
        losses['ae'] = ae_loss.item()
        

        
        # Discriminate
        #
        batch_pos_neg = torch.cat([ground_truth, imgs_complete], dim=0) # [N3HW]
        
        # Add in mask and repeat for ground truth and generated completion.
        # Will be split later to produce "real" and "fake" patch features in discriminator
        # for use with hinge loss.
        #
        batch_pos_neg= torch.cat([batch_pos_neg, mask_ff.repeat(2, 1, 1, 1)], dim=1) # [N4HW]
#         patchD.zero_grad()
        pos_neg = patchD(batch_pos_neg)
        
        
        # Losses
        #
        pos, neg = torch.chunk(pos_neg, 2, dim=0)
        g_loss, d_loss = gan_hinge_loss(pos, neg)
        g_loss += ae_loss
        losses['g_loss'] = g_loss.item()
        losses['d_loss'] = d_loss.item()
        
        
        compute_g_loss = iteration % config['n_critic'] == 0
#         # Optimize
#         #
#         if not compute_g_loss:
        optimizer_d.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer_d.step()
        
        
        pos_neg = patchD(batch_pos_neg)
        pos, neg = torch.chunk(pos_neg, 2, dim=0)
        g_loss, d_loss = gan_hinge_loss(pos, neg)
        g_loss += ae_loss

#         if compute_g_loss:
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        

#         print("ae_loss: ", ae_loss, " g_loss: ", g_loss, " d_loss: ", d_loss)
        # Set tqdm description
        #
        if local_rank == 0:
            message = ' '
            for k in losses:
#                 v = losses.get(k, 0.)
                v = losses[k]
#                 writer.add_scalar(k, v, iteration)
                message += '%s: %.4f ' % (k, v)

            pbar.set_description(
                (
                    f" {message}"
                )
            )
        
        # Save output from current iteration.
        #
        if local_rank == 0:      
            if iteration % (config['viz_iter']) == 0:
                    viz_max_out = config['viz_max_out']
                    if ground_truth.size(0) > viz_max_out:
                        viz_images = torch.stack(
                            [ground_truth[:viz_max_out],
                             imgs_incomplete[:viz_max_out],
                             imgs_complete[:viz_max_out],
                             offset_flow[:viz_max_out]],
                             dim=1
                        )
                    else:
                        viz_images = torch.stack(
                            [ground_truth,
                             imgs_incomplete,
                             imgs_complete,
                             offset_flow],
                             dim=1
                        )
                    viz_images = viz_images.view(-1, *list(ground_truth.size())[1:])
                    vutils.save_image(viz_images,
                                      '%s/niter_%08d.png' % (checkpoint_path, iteration),
                                      nrow=2 * 4,
                                      normalize=True)
        
            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                save_model_v2(netG, patchD, optimizer_g, optimizer_d, checkpoint_path, iteration)
        
        
        
        
        
        
def train_distributed(config, logger, writer, checkpoint_path):
    
    dist.init_process_group(                                   
        backend='nccl',
#         backend='gloo',
        init_method='env://'
    )  
    
    
    # Find out what GPU on this compute node.
    #
    local_rank = torch.distributed.get_rank()
    
    
    # this is the total # of GPUs across all nodes
    # if using 2 nodes with 4 GPUs each, world size is 8
    #
    world_size = torch.distributed.get_world_size()
    print("### global rank of curr node: {} of {}".format(local_rank, world_size))
    
    
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    #
    print("local_rank: ", local_rank)
#     dist.barrier()
    torch.cuda.set_device(local_rank)
    
    
    # Define the trainer
    print("Creating models on device: ", local_rank)
    
    
    input_dim = config['netG']['input_dim']
    cnum = config['netG']['ngf']
    use_cuda = True
    gated = config['netG']['gated']
    
    
    # Models
    #
    netG = Generator(config['netG'], use_cuda=True, device=local_rank).cuda()
    netG = torch.nn.parallel.DistributedDataParallel(
        netG,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    
    localD = LocalDis(config['netD'], use_cuda=True, device_id=local_rank).cuda()
    localD = torch.nn.parallel.DistributedDataParallel(
        localD,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    
    
    globalD = GlobalDis(config['netD'], use_cuda=True, device_id=local_rank).cuda()
    globalD = torch.nn.parallel.DistributedDataParallel(
        globalD,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    
    
    if local_rank == 0:
        logger.info("\n{}".format(netG))
        logger.info("\n{}".format(localD))
        logger.info("\n{}".format(globalD))
        
    
    # Optimizers
    #
    optimizer_g = torch.optim.Adam(
        netG.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2'])
    )

    
    d_params = list(localD.parameters()) + list(globalD.parameters())
    optimizer_d = torch.optim.Adam(
        d_params,  
        lr=config['lr'],                                    
        betas=(config['beta1'], config['beta2'])                              
    )
    
    
    # Data
    #
    sampler = None
    train_dataset = Dataset(
        data_path=config['train_data_path'],
        with_subfolder=config['data_with_subfolder'],
        image_shape=config['image_shape'],
        random_crop=config['random_crop']
    )
        
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
#             num_replicas=torch.cuda.device_count(),
        num_replicas=len(config['gpu_ids']),
#         rank = local_rank
    )
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=(sampler is None),
        num_workers=config['num_workers'],
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    
    
    # Get the resume iteration to restart training
    #
#     start_iteration = trainer.resume(config['resume']) if config['resume'] else 1
    start_iteration = 1
    print("\n\nStarting epoch: ", start_iteration)

    iterable_train_loader = iter(train_loader)

    if local_rank == 0: 
        time_count = time.time()

    epochs = config['niter'] + 1
    pbar = tqdm(range(start_iteration, epochs), dynamic_ncols=True, smoothing=0.01)
    for iteration in pbar:
        sampler.set_epoch(iteration)
        
        try:
            ground_truth = next(iterable_train_loader)
        except StopIteration:
            iterable_train_loader = iter(train_loader)
            ground_truth = next(iterable_train_loader)

        # Prepare the inputs
        bboxes = random_bbox(config, batch_size=ground_truth.size(0))
        x, mask = mask_image(ground_truth, bboxes, config)

        
        # Move to proper device.
        #
        bboxes = bboxes.cuda(local_rank)
        x = x.cuda(local_rank)
        mask = mask.cuda(local_rank)
        ground_truth = ground_truth.cuda(local_rank)
        

        ###### Forward pass ######
        compute_g_loss = iteration % config['n_critic'] == 0
#         losses, inpainted_result, offset_flow = forward(config, x, bboxes, mask, ground_truth,
#                                                        localD=localD, globalD=globalD,
#                                                        coarse_gen=coarse_generator, fine_gen=fine_generator,
#                                                        local_rank=local_rank, compute_loss_g=compute_g_loss)
        losses, inpainted_result, offset_flow = forward(config, x, bboxes, mask, ground_truth,
                                                       netG=netG, localD=localD, globalD=globalD,
                                                       local_rank=local_rank, compute_loss_g=compute_g_loss)

        
        # Scalars from different devices are gathered into vectors
        #
        for k in losses.keys():
            if not losses[k].dim() == 0:
                losses[k] = torch.mean(losses[k])
                
                
        ###### Backward pass ######
        # Update D
        if not compute_g_loss:
            optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            optimizer_d.step() 

        # Update G
        if compute_g_loss:
            optimizer_g.zero_grad()
            losses['g'] = losses['ae'] * config['ae_loss_alpha']
            losses['g'] += losses['l1'] * config['l1_loss_alpha']
            losses['g'] += losses['wgan_g'] * config['gan_loss_alpha']
            losses['g'].backward()
            optimizer_g.step()


        # Set tqdm description
        #
        if local_rank == 0:
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            message = ' '
            for k in log_losses:
                v = losses.get(k, 0.)
                writer.add_scalar(k, v, iteration)
                message += '%s: %.4f ' % (k, v)

            pbar.set_description(
                (
                    f" {message}"
                )
            )
            
                
        if local_rank == 0:      
            if iteration % (config['viz_iter']) == 0:
                    viz_max_out = config['viz_max_out']
                    if x.size(0) > viz_max_out:
                        viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                                  offset_flow[:viz_max_out]], dim=1)
                    else:
                        viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                    viz_images = viz_images.view(-1, *list(x.size())[1:])
                    vutils.save_image(viz_images,
                                      '%s/niter_%08d.png' % (checkpoint_path, iteration),
                                      nrow=3 * 4,
                                      normalize=True)

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                save_model(
                    netG, globalD, localD, optimizer_g, optimizer_d, checkpoint_path, iteration
                )
                

           
                
                
                
def train(config, logger, checkpoint_path):
    try:  # for unexpected error logging
        # Load the dataset
        logger.info("Training on dataset: {}".format(config['dataset_name']))
        train_dataset = Dataset(data_path=config['train_data_path'],
                                with_subfolder=config['data_with_subfolder'],
                                image_shape=config['image_shape'],
                                random_crop=config['random_crop'])
        # val_dataset = Dataset(data_path=config['val_data_path'],
        #                       with_subfolder=config['data_with_subfolder'],
        #                       image_size=config['image_size'],
        #                       random_crop=config['random_crop'])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        #                                           batch_size=config['batch_size'],
        #                                           shuffle=False,
        #                                           num_workers=config['num_workers'])

        # Define the trainer
        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.localD))
        logger.info("\n{}".format(trainer.globalD))

        
#         if cuda:
#             trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
#             trainer_module = trainer.module
#         else:
#             trainer_module = trainer
        trainer_module = trainer

            
        # Get the resume iteration to restart training
        #
        start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1
        print("\n\nStarting epoch: ", start_iteration)

        iterable_train_loader = iter(train_loader)

    
        time_count = time.time()


        epochs = config['niter'] + 1
        pbar = tqdm(range(start_iteration, epochs), dynamic_ncols=True, smoothing=0.01)
#         for iteration in range(start_iteration, epochs):
        for iteration in pbar:
            try:
                ground_truth = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                ground_truth = next(iterable_train_loader)

            # Prepare the inputs
            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            x, mask = mask_image(ground_truth, bboxes, config)
            x = x.cuda()
            mask = mask.cuda()
            ground_truth = ground_truth.cuda()

            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0
            losses, inpainted_result, offset_flow = trainer(x, bboxes, mask, ground_truth, compute_g_loss)
            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            if not compute_g_loss:
                trainer_module.optimizer_d.zero_grad()
                losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
                losses['d'].backward()
                trainer_module.optimizer_d.step()

            # Update G
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                              + losses['ae'] * config['ae_loss_alpha'] \
                              + losses['wgan_g'] * config['gan_loss_alpha']
                losses['g'].backward()
                trainer_module.optimizer_g.step()

                
            ### TODO:
            ### - Why does this need to be moved from above to here?
            ###
#             losses['d'].backward()
#             trainer_module.optimizer_d.step()    
            
            
            # Set tqdm description
            #
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']  
#             message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
            message = ' '
            for k in log_losses:
                v = losses.get(k, 0.)
                writer.add_scalar(k, v, iteration)
                message += '%s: %.4f ' % (k, v)
                
            pbar.set_description(
                (
                    f" {message}"
                )
            )
                
                
            # Log and visualization
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses.get(k, 0.)
                    writer.add_scalar(k, v, iteration)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
#                 logger.info(message)
                

            if iteration % (config['viz_iter']) == 0:
                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                              offset_flow[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                  nrow=3 * 4,
                                  normalize=True)

                
            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)
    
    
    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e
        
        


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    
    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
#         os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

        
    # datetime object containing current date and time
    #
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H.%M.%S") 
    
    
    # Configure checkpoint path
    checkpoint_path = os.path.join(
        'ckpts',
        config['dataset_name'] + '_' + dt_string,
        config['mask_type'] + '_' + config['expname']
    )

    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
        
        
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(logdir=checkpoint_path)
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call
    logger.info("Arguments: {}".format(args))
    
    
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

        
    # Log the configuration
    logger.info("Configuration: {}".format(config))
    
    if args.distributed:
        print("Distributed training...")
#         train_distributed(config, logger, writer, checkpoint_path)
        train_distributed_v2(config, logger, writer, checkpoint_path)
        
    else:
        train(config, logger, checkpoint_path)
        


if __name__ == '__main__':
    main()

    

# Example training run:
# -------------------------------
#
# Distributed 1 Node, 1 GPU:
# -----------------
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train.py --config=configs/config-gated-spectnorm.yaml --distributed
#
# Distributed 1 Node, 4 GPU:
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py --config=configs/config-gated-spectnorm-freeform.yaml --distributed

# Distributed 1 Node, 8 GPU:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 train.py --config=configs/config-gated-spectnorm-freeform.yaml --distributed