import datetime
import logging
import os
import random
import time
from contextlib import nullcontext
from glob import glob
from functools import partial
from collections import defaultdict
import pickle

import matplotlib
import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import Value
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import wandb

import datasets
from datasets.fingerprint import Hasher
from datasets import load_dataset_builder
import eqnet
import utils
from eqnet.data import (
    AutoEncoderIterableDataset,
    DASIterableDataset,
    SeismicNetworkIterableDataset,
    SeismicTraceIterableDataset,
)
from eqnet.models.unet import moving_normalize
from eqnet.utils.station_sampler import StationSampler, create_groups, cut_reorder_keys, reorder_keys

os.system('export HF_DATASETS_CACHE="/pscratch/sd/w/whaoyu/huggingface"')
matplotlib.use("agg")
logger = logging.getLogger("EQNet")
# mp.set_start_method("spawn", force=True)

from train import evaluate, train_one_epoch, plot_results, get_args_parser

def main(args):
    # mp.set_start_method('spawn', force=True)
    if args.output_dir:
        utils.mkdir(args.output_dir)
        figure_dir = os.path.join(args.output_dir, "figures")
        args.figure_dir = figure_dir
        utils.mkdir(figure_dir)
        
    if args.huggingface_cache is not None:
        torch.set_num_threads(1)
        data_path = f"{os.path.dirname(args.hdf5_file)}/waveform_h5"
        code_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = os.path.join(code_dir, "eqnet/data/quakeflow_nc.py")
        print(f"loading huggingface dataset from {data_path} with script {script_dir}")
        dataset_builder = load_dataset_builder(script_dir, name='event_custom', cache_dir=args.huggingface_cache, data_files={"train": sorted(glob(f"{data_path}/**", recursive=False))}, trust_remote_code=True)
        dataset_builder.download_and_prepare(file_format="arrow", num_proc=16)

    utils.init_distributed_mode(args)
    print(args)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
        print(f"Rank: {rank}, World size: {world_size}")
    else:
        rank, world_size = 0, 1

    torch.manual_seed(1337 + rank)
    random.seed(1337 + rank)
    np.random.seed(1337 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337 + rank)

    device = torch.device(args.device)
    dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    scaler = torch.amp.GradScaler('cuda', enabled=((dtype == "float16") & torch.cuda.is_available()))
    args.dtype, args.ptdtype = dtype, ptdtype
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.model in ["phasenet", "phasenet_plus"]:
        dataset = SeismicTraceIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            hdf5_file=args.hdf5_file,
            format="h5",
            training=True,
            stack_event=args.stack_event,
            flip_polarity=args.flip_polarity,
            drop_channel=args.drop_channel,
            rank=rank,
            world_size=world_size,
        )
        train_sampler = None
        dataset_test = SeismicTraceIterableDataset(
            data_path=args.test_data_path,
            data_list=args.test_data_list,
            hdf5_file=args.test_hdf5_file,
            format="h5",
            training=True,
            stack_event=False,
            flip_polarity=False,
            drop_channel=False,
            rank=rank,
            world_size=world_size,
        )
        test_sampler = None
    elif args.model == "phasenet_das":
        dataset = DASIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            label_path=args.label_path,
            label_list=args.label_list,
            noise_list=args.noise_list,
            phases=args.phases,
            nt=args.nt,
            nx=args.nx,
            format="h5",
            training=True,
            stack_noise=args.stack_noise,
            stack_event=args.stack_event,
            resample_space=args.resample_space,
            resample_time=args.resample_time,
            masking=args.masking,
            rank=rank,
            world_size=world_size,
        )
        train_sampler = None
        dataset_test = DASIterableDataset(
            data_path=args.test_data_path,
            data_list=args.test_data_list,
            label_path=args.test_label_path,
            label_list=args.test_label_list,
            noise_list=args.test_noise_list,
            phases=args.phases,
            nt=args.nt,
            nx=args.nx,
            format="h5",
            training=True,
            stack_noise=False,
            stack_event=False,
            resample_space=False,
            resample_time=False,
            masking=False,
            rank=rank,
            world_size=world_size,
        )
        test_sampler = None
    elif args.model == "autoencoder":
        dataset = AutoEncoderIterableDataset(
            data_path=args.data_path,
            format="h5",
            training=True,
        )
        train_sampler = None
        dataset_test = dataset
        test_sampler = None
    elif args.model == "eqnet":
        if args.huggingface_dataset:
            # Get the directory of the train.py
            code_dir = os.path.dirname(os.path.abspath(__file__))
            script_dir = os.path.join(code_dir, "eqnet/data/quakeflow_nc.py")
            # dataset_builder = load_dataset_builder(script_dir, name=args.dataset_config, cache_dir="/data/cache/huggingface")
            # dataset_builder.download_and_prepare(file_format="arrow", num_proc=min(args.workers, 16))
            # dataset = dataset_builder.as_dataset(split="train")
            # dataset_test = dataset_builder.as_dataset(split="test")
            dataset = datasets.load_dataset(script_dir, split="train", name=args.dataset_config, cache_dir="/data/cache/huggingface", num_proc=min(args.workers, 16))
            dataset_test = datasets.load_dataset(script_dir, split="test", name=args.dataset_config, cache_dir="/data/cache/huggingface", num_proc=1)
            dataset = dataset.with_format("torch")
            dataset_test = dataset_test.with_format("torch")
            if args.distributed and world_size > 1:
                torch.set_num_threads(1) # fix the multi-processing bug
                if args.gpu > 0:
                    print(f"Rank {args.rank}: Gpu {args.gpu} waiting for main process to perform the filtering", force=True)
                    torch.distributed.barrier()
                # we discard the samples with less than 5 stations
                dataset = dataset.filter(lambda x: x["num_stations"] >= 5, num_proc=args.workers)
                # dataset_test = dataset_test.map(lambda x: reorder_keys(x), num_proc=args.workers, desc="reorder_keys")
                if args.gpu == 0:
                    print("Filtering finished, loading results from main process")
                    torch.distributed.barrier()
            else:
                torch.set_num_threads(1) # fix the multi-processing bug
                print("Filtering dataset")
                # we discard the samples with less than 5 stations
                dataset = dataset.filter(lambda x: x["num_stations"] >= 5, num_proc=args.workers)
            
            group_ids = create_groups(dataset, args.num_stations_list, is_pad=True)
            cut_reorder_keys_group = partial(cut_reorder_keys, num_stations_list=args.num_stations_list)
            if args.distributed and world_size > 1:
                torch.set_num_threads(1) # fix the multi-processing bug
                if args.gpu > 0:
                    print(f"Rank {args.rank}: Gpu {args.gpu} waiting for main process to perform the mapping", force=True)
                    torch.distributed.barrier()
                print(f"Rank {args.rank}: Gpu {args.gpu} cut_reorder_keys fingerprint {Hasher.hash(lambda x: cut_reorder_keys_group(x, is_pad=True))}", force=True)
                dataset = dataset.map(lambda x: cut_reorder_keys_group(x, is_pad=True), num_proc=args.workers, desc="cut_reorder_keys")#, new_fingerprint=Hasher.hash(lambda x: cut_reorder_keys(x, num_stations_list=num_stations_list, is_pad=True, is_train=True)))
                dataset_test = dataset_test.map(lambda x: cut_reorder_keys_group(x, is_pad=True), num_proc=args.workers, desc="cut_reorder_keys")#, new_fingerprint=Hasher.hash(lambda x: cut_reorder_keys(x, num_stations_list=num_stations_list, is_pad=True, is_train=False)))
                # dataset_test = dataset_test.map(lambda x: reorder_keys(x), num_proc=args.workers, desc="reorder_keys")
                if args.gpu == 0:
                    print("Mapping finished, loading results from main process")
                    torch.distributed.barrier()
            else:
                torch.set_num_threads(1) # fix the multi-processing bug
                print("Mapping dataset")
                dataset = dataset.map(lambda x: cut_reorder_keys(x, num_stations_list=args.num_stations_list, is_pad=True), desc="cut_reorder_keys", num_proc=args.workers)
                dataset_test = dataset_test.map(lambda x: cut_reorder_keys(x, num_stations_list=args.num_stations_list, is_pad=True), desc="cut_reorder_keys", num_proc=args.workers)

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
            else:
                train_sampler = torch.utils.data.RandomSampler(dataset)
                test_sampler = torch.utils.data.SequentialSampler(dataset_test)

            train_batch_sampler = StationSampler(train_sampler, group_ids, args.batch_size, args.num_stations_list)

        else:
            dataset = SeismicNetworkIterableDataset(
                data_path=args.data_path,
                data_list=args.data_list,
                hdf5_file=args.hdf5_file,
                num_stations=args.num_stations_list[0] if args.num_stations_list is not None else None,
                format="h5",
                training=True,
                stack_event=args.stack_event,
                concatenate_event=args.cat_event,
                drop_channel=args.drop_channel,
                sample_by_magnitude=args.sample_by_magnitude,
                sample_by_distance=args.sample_by_distance,
                huggingface_cache=args.huggingface_cache,
                attribute_cache=args.attr_cache,
                rank=rank,
                world_size=world_size,
            )    
            train_sampler = None
            dataset_test = SeismicNetworkIterableDataset(
                data_path=args.test_data_path,
                data_list=args.test_data_list,
                hdf5_file=args.test_hdf5_file,
                num_stations=args.num_stations_list[0] if args.num_stations_list is not None else None,
                format="h5",
                training=True,
                stack_event=False,
                drop_channel=False,
                rank=rank,
                world_size=world_size,
            )  
            test_sampler = None
            
            negative_mask_ratio = Value("d", 1.0)
            cat_event_ratio = Value("d", 0.08)
            def worker_init_fn(negative_mask_ratio, cat_event_ratio, worker_id):
                info = torch.utils.data.get_worker_info()
                info.dataset.negative_mask_ratio = negative_mask_ratio
                info.dataset.cat_event_ratio = cat_event_ratio
                #print(f"Worker {worker_id} negative_mask_ratio: {negative_mask_ratio}")
    else:
        raise NotImplementedError(f"Invalid model {args.model}. Only phasenet, phasenet_das, autoencoder and eqnet are supported.")

    if args.huggingface_dataset:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.workers,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            sampler=test_sampler,
            num_workers=args.workers,
            collate_fn=None,
            drop_last=False,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            # collate_fn=utils.collate_fn,
            collate_fn=None,
            # shuffle=True,
            drop_last=True,
            prefetch_factor=5,
            #pin_memory=True,
            persistent_workers=True,
            worker_init_fn=partial(worker_init_fn, negative_mask_ratio, cat_event_ratio) if args.model == "eqnet" else None,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            sampler=test_sampler,
            num_workers=args.workers,
            collate_fn=None,
            drop_last=False,
        )

    model = eqnet.models.__dict__[args.model].build_model(
        backbone=args.backbone,
        in_channels=1,
        out_channels=(len(args.phases) + 1),
        ## eqnet
        head=args.head,
        use_station_location=args.use_station_location,
    )
    logger.info("Model:\n{}".format(model))

    print("Model:\n{}".format(model))

    model.to(device)
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)  # requires PyTorch 2.0

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # logging
    if args.wandb and utils.is_main_process():
        wandb.init(
            project=args.wandb_project if args.wandb_project else args.model,
            name=args.wandb_name,
            entity=args.wandb_group,
            dir=args.wandb_dir,
            config=args,
        )
        if args.wandb_watch:
            wandb.watch(model, log="all", log_freq=args.print_freq)

    if args.resume_wandb:
        if utils.is_main_process():
            if args.model == "phasenet_das":
                model_url = "ai4eps/model-registry/PhaseNet-DAS:latest"
                artifact = wandb.use_artifact(model_url, type="model")
                artifact_dir = artifact.download()
                checkpoint = torch.load(glob(os.path.join(artifact_dir, "*.pth"))[0], map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=True)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.custom_weight_decay is not None:
        #NOTE: actually no weight decay
        assert isinstance(args.custom_weight_decay, float), f'use same weight decay for all custom keys or hack this code'
        if hasattr(model, 'no_weight_decay'):
            skip_list = model.no_weight_decay()
            for k in skip_list:
                custom_keys_weight_decay.append((k, args.custom_weight_decay))
        if hasattr(model, 'no_weight_decay_keywords'):
            skip_list = model.no_weight_decay_keywords()
            for k in skip_list:
                custom_keys_weight_decay.append((k, args.custom_weight_decay))
    
    
    ## warmup stage
    for param in model.parameters():
        param.requires_grad = False
    for param in model.phase_associator.parameters():
        param.requires_grad = True
    model.apply(lambda m: m.eval() if isinstance(m, nn.modules.batchnorm._BatchNorm) else None)
    model.phase_associator.apply(lambda m: m.train() if isinstance(m, nn.modules.batchnorm._BatchNorm) else None)
    warmup_parameters = utils.set_weight_decay(
        model.phase_associator,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    optimizer = torch.optim.AdamW(warmup_parameters, lr=args.new_warmup_lr, weight_decay=args.weight_decay)
    # constant lr schedule
    iters_per_epoch = len(data_loader)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=iters_per_epoch*args.new_warmup_epoch)
    tasks = ["loss_focal", "loss_dice", "loss_iou"]
    pre_weight = list(model.phase_associator.weights.values())
    print(f"{args.model}: constant pre_weight: ", end="\t")
    for i, k in enumerate(tasks):
        print(f"{k}: {pre_weight[i]:.4f}", end="\t")
    print()
    warmup_balancer = utils.MTLBalance('constant', tasks, pre_weight, args.temp, args.epochs+1, device)

    model_without_ddp = model
    # print parameter number
    #print("Number of parameters: ", sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad))
    backbone_params = sum(p.numel() for p in model_without_ddp.backbone.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f"Toal parameters: {total_params}, backbone parameters: {backbone_params}, other parameters: {total_params - backbone_params}")
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(name)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )  # , find_unused_parameters=True)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    base_epoch = 0
    if args.pretrained:
        backbone_pretrained = None
        if os.path.isfile(args.pretrained):
            print(f"Loading pretrained model from '{args.pretrained}'")
            backbone_pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        elif os.path.isfile(args.output_dir + "/backbone.pth"):
            print(f"Loading pretrained model from '{args.output_dir}/backbone.pth'")
            backbone_pretrained = torch.load(args.output_dir + "/backbone.pth", map_location="cpu", weights_only=False)
        else:
            print(f"No pretrained model found")
        
        if backbone_pretrained is not None:
            base_epoch = backbone_pretrained.get("epoch", -1) + 1
            if "model" in backbone_pretrained:
                model_without_ddp.load_state_dict(backbone_pretrained["model"], strict=False)
            else:
                model_without_ddp.load_state_dict(backbone_pretrained, strict=False)
    if args.warmup_resume or args.resume:
        checkpoint = None
        if isinstance(args.warmup_resume, str) and os.path.isfile(args.warmup_resume):
            print(f"Loading model and optimizer from checkpoint '{args.warmup_resume}'")
            checkpoint = torch.load(args.warmup_resume, map_location="cpu", weights_only=False)
        elif os.path.isfile(args.output_dir + "/checkpoint.pth"):
            print(f"Loading model and optimizer from checkpoint '{args.output_dir}/checkpoint.pth'")
            checkpoint = torch.load(args.output_dir + "/checkpoint.pth", map_location="cpu", weights_only=False)
        else:
            print(f"No checkpoint found")
        if checkpoint is not None:
            if (checkpoint.get("stage", "warmup") != "warmup" or checkpoint["epoch"] == args.new_warmup_epoch-1) \
                and (not args.warmup_resume):
                print(f"Skip loading full training checkpoint in warmup stage")
            else:
                args.warmup_resume = True # automatically detect warmup resume
                model_without_ddp.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
                if model_ema:
                    model_ema.load_state_dict(checkpoint["model_ema"])
                if scaler:
                    scaler.load_state_dict(checkpoint["scaler"])
    
    print("Warm up training for phase_associator")
    start_time = time.time()
    best_loss = float("inf")
    for epoch in range(args.new_warmup_epoch):
        if args.resume and (not args.warmup_resume):
            print("Skip warmup stage when resume training from full training checkpoint")
            break
        if epoch>args.new_warmup_epoch//2 and args.model == "eqnet" and not args.huggingface_dataset:
            with cat_event_ratio.get_lock():
                cat_event_ratio.value = 0.12 if epoch>args.new_warmup_epoch//2 else 0.08
        if args.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)

        tmp_time = time.time()
        train_one_epoch(
            model,
            optimizer,
            lr_scheduler,
            data_loader,
            model_ema,
            scaler,
            args,
            epoch,
            len(dataset),
            balancer=warmup_balancer,
            freeze_batchnorm=True,
        )
        print(f"Training time of epoch {epoch} of rank {rank}: {time.time() - tmp_time:.3f}")

        #if args.test_hdf5_file is not None:
        #    tmp_time = time.time()
        #    metric = evaluate(model, data_loader_test, scaler, args, epoch, len(dataset_test))
        #    if model_ema:
        #        metric = evaluate(model_ema, data_loader_test, scaler, args, epoch, len(dataset_test))
        #    print(f"Testing time of epoch {epoch} of rank {rank}: {time.time() - tmp_time:.3f}")

        tmp_time = time.time()
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "stage": "warmup",# if epoch < args.new_warmup_epoch -1 else "full",
        }
        if model_ema:
            checkpoint["model_ema"] = model_ema.state_dict()
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()

        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_warmup_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        #if utils.is_main_process() and metric.loss.global_avg < best_loss:
        #    best_loss = metric.loss.global_avg
        #    torch.save(checkpoint, os.path.join(args.output_dir, "model_best.pth"))
            #if args.wandb:
            #    best_model = wandb.Artifact(
            #        f"{args.wandb_name}",
            #        type="model",
            #        metadata=dict(epoch=epoch, loss=best_loss),
            #    )
            #    best_model.add_file(os.path.join(args.output_dir, "model_best.pth"))
            #    wandb.log_artifact(best_model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    ## main stage
    original_model = model.module if args.distributed else model
    for param in original_model.parameters():
        param.requires_grad = True    
        
    other_modules = torch.nn.ModuleList([m for n, m in original_model.named_children() if n != "phase_associator"])
    main_parameters = utils.set_weight_decay(
        other_modules,
        0.0, # no weight decay for backbone and other modules
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    for group in main_parameters:
        group['lr'] = args.lr
    new_parameters = utils.set_weight_decay(
        original_model.phase_associator,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    for group in new_parameters:
        group['lr'] = args.new_full_lr
    parameters = main_parameters + new_parameters

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    iters_per_epoch = len(data_loader)
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size * iters_per_epoch, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.epochs - args.lr_warmup_epochs) * iters_per_epoch, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "polynomiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and PolynomialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs * iters_per_epoch
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs * iters_per_epoch
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs * iters_per_epoch],
        )
    else:
        lr_scheduler = main_lr_scheduler
        
    if args.mtl is not None:
        if args.mtl == "dwa" or args.mtl == "dwa_inverse":
            if args.model == "eqnet":
                #tasks = ["loss_phase", "loss_event", "loss_offset", "loss_hypocenter"]
                tasks = ["loss_phase", "loss_event", "loss_offset", "loss_origin_time", "loss_x", "loss_y", "loss_depth", "loss_magnitude", "loss_polarity", "loss_focal", "loss_dice", "loss_iou"]
                # pre_weight = args.pre_weight
                pre_weight = original_model.phase_picker.weights+original_model.event_detector.weights+original_model.polarity_picker.weights + list(original_model.phase_associator.weights.values())
                print(f"{args.model}: {args.mtl} pre_weight: ", end="\t")
                for i, k in enumerate(tasks):
                    print(f"{k}: {pre_weight[i]:.4f}", end="\t")
                print()
        
        balancer = utils.MTLBalance(args.mtl, tasks, pre_weight, args.temp, args.epochs+1, device)

    model_without_ddp = original_model
    # print parameter number
    #print("Number of parameters: ", sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad))
    backbone_params = sum(p.numel() for p in model_without_ddp.backbone.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f"Toal parameters: {total_params}, backbone parameters: {backbone_params}, other parameters: {total_params - backbone_params}")
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(name)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            original_model, device_ids=[args.gpu]
        )  # , find_unused_parameters=True)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
        
    del data_loader
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        # collate_fn=utils.collate_fn,
        collate_fn=None,
        # shuffle=True,
        drop_last=True,
        #prefetch_factor=5,
        #pin_memory=True,
        persistent_workers=True,
        worker_init_fn=partial(worker_init_fn, negative_mask_ratio, cat_event_ratio) if args.model == "eqnet" else None,
    )
    
    if args.resume:
        checkpoint = None
        if isinstance(args.resume, str) and os.path.isfile(args.resume):
            print(f"Loading model and optimizer from checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        elif os.path.isfile(args.output_dir + "/checkpoint.pth"):
            print(f"Loading model and optimizer from checkpoint '{args.output_dir}/checkpoint.pth'")
            checkpoint = torch.load(args.output_dir + "/checkpoint.pth", map_location="cpu", weights_only=False)
        else:
            print(f"No checkpoint found")
        if checkpoint is not None:
            model_without_ddp.load_state_dict(checkpoint["model"])
            args.start_epoch = checkpoint["epoch"] + 1 - 33 if checkpoint['stage'] != "warmup" else 0
            if checkpoint['stage'] != "warmup":
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            base_epoch = 33
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    best_loss = float("inf")
    for epoch in range(args.start_epoch+base_epoch, args.epochs+base_epoch):
        if epoch%5==0 and epoch >= 5 and args.model == "eqnet" and not args.huggingface_dataset:
            #negative_mask_ratio = max(1.0 - 0.03 * epoch, 0.3)
            with negative_mask_ratio.get_lock():
                negative_mask_ratio.value = max(1.0 - 0.006 * epoch, 0.4)
        if epoch>args.epochs//3 and args.model == "eqnet" and not args.huggingface_dataset:
            with cat_event_ratio.get_lock():
                cat_event_ratio.value = 0.12 if epoch<2*args.epochs//3 else 0.16
        if args.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)

        tmp_time = time.time()
        train_one_epoch(
            model,
            optimizer,
            lr_scheduler,
            data_loader,
            model_ema,
            scaler,
            args,
            epoch,
            len(dataset),
            balancer=None if args.mtl is None else balancer,
            freeze_batchnorm=True,
        )
        print(f"Training time of epoch {epoch} of rank {rank}: {time.time() - tmp_time:.3f}")

        if args.test_hdf5_file is not None:
            tmp_time = time.time()
            metric = evaluate(model, data_loader_test, scaler, args, epoch, len(dataset_test))
            if model_ema:
                metric = evaluate(model_ema, data_loader_test, scaler, args, epoch, len(dataset_test))
            print(f"Testing time of epoch {epoch} of rank {rank}: {time.time() - tmp_time:.3f}")

        tmp_time = time.time()
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "stage": "full",
        }
        if model_ema:
            checkpoint["model_ema"] = model_ema.state_dict()
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()

        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        if utils.is_main_process() and metric.loss.global_avg < best_loss:
            best_loss = metric.loss.global_avg
            torch.save(checkpoint, os.path.join(args.output_dir, "model_best.pth"))
            #if args.wandb:
            #    best_model = wandb.Artifact(
            #        f"{args.wandb_name}",
            #        type="model",
            #        metadata=dict(epoch=epoch, loss=best_loss),
            #    )
            #    best_model.add_file(os.path.join(args.output_dir, "model_best.pth"))
            #    wandb.log_artifact(best_model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if args.wandb and utils.is_main_process():
        wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--new-warmup-epoch", default=3, type=int, help="warm up epochs for new component")
    parser.add_argument("--new-warmup-lr", default=1e-3, type=float, help="learning rate for new component in warm up and full training")
    parser.add_argument("--new-full-lr", default=2e-4, type=float, help="full learning rate for new component")
    parser.add_argument("--new-warmup-method", default="constant", type=str, help="warm up method for new component")
    parser.add_argument("--warmup-resume", default=None, type=str, help="resume from warmup checkpoint")
    args = parser.parse_args()
    main(args)
