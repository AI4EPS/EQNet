import datetime
import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision

import eqnet
import utils
from eqnet.data import (
    AutoEncoderIterableDataset,
    DASDataset,
    DASIterableDataset,
    SeismicNetworkIterableDataset,
    SeismicTraceIterableDataset,
)
from eqnet.models import log_transform, normalize_local

matplotlib.use("agg")

logger = logging.getLogger("EQNet")


def evaluate(model, data_loader, device, num_classes=3, args=None):
    model.eval()
    # confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        i = 0
        for meta in metric_logger.log_every(data_loader, 100, header):

            if args.model == "phasenet":
                out = model(meta)
                phase = F.softmax(out["phase"], dim=1).cpu()
                event = torch.sigmoid(out["event"]).cpu()
                print("Plotting...")
                eqnet.utils.visualize_phasenet_train(meta, phase, event, epoch=i, figure_dir=args.output_dir)
                del phase, event

            if args.model == "deepdenoiser":
                pass

            elif args.model == "phasenet_das":
                preds = model(meta)
                preds = F.softmax(preds, dim=1).cpu()
                print("Plotting...")
                eqnet.utils.visualize_das_train(meta, preds, epoch=i, figure_dir=args.output_dir)
                del preds

            elif args.model == "autoencoder":
                preds = model(meta).cpu()
                print("Plotting...")
                eqnet.utils.visualize_autoencoder_das_train(meta, preds, epoch=i, figure_dir=args.output_dir)
                del preds

            elif args.model == "eqnet":
                out = model(meta)
                phase = F.softmax(out["phase"], dim=1).cpu()
                event = torch.sigmoid(out["event"]).cpu()
                print("Plotting...")
                eqnet.utils.visualize_eqnet_train(meta, phase, event, epoch=i, figure_dir=args.output_dir)
                del phase, event

            i += 1
            # confmat.update(meta["targets"].argmax(1).flatten(), output.argmax(1).flatten().cpu())

        # confmat.reduce_from_all_processes()

    # return confmat
    return


def train_one_epoch(
    model, optimizer, data_loader, lr_scheduler, device, epoch, iters_per_epoch, print_freq, scaler=None, args=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    i = 0
    for meta in metric_logger.log_every(data_loader, print_freq, header):

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss = model(meta)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        i += 1
        # if i > len(data_loader):
        if i > 200:
            break
        if i > iters_per_epoch:
            break
        # break

    model.eval()
    with torch.inference_mode():

        if args.model == "phasenet":
            out = model(meta)
            phase = F.softmax(out["phase"], dim=1).cpu()
            event = torch.sigmoid(out["event"]).cpu()
            print("Plotting...")
            eqnet.utils.visualize_phasenet_train(meta, phase, event, epoch=epoch, figure_dir=args.output_dir)
            del phase, event

        if args.model == "deepdenoiser":
            pass

        elif args.model == "phasenet_das":
            preds = model(meta)
            preds = F.softmax(preds, dim=1).cpu()
            print("Plotting...")
            eqnet.utils.visualize_das_train(meta, preds, epoch=epoch, figure_dir=args.output_dir)
            del preds

        elif args.model == "autoencoder":
            preds = model(meta).cpu()
            print("Plotting...")
            eqnet.utils.visualize_autoencoder_das_train(meta, preds, epoch=epoch, figure_dir=args.output_dir)
            del preds

        elif args.model == "eqnet":
            out = model(meta)
            phase = F.softmax(out["phase"], dim=1).cpu()
            event = torch.sigmoid(out["event"]).cpu()
            print("Plotting...")
            eqnet.utils.visualize_eqnet_train(meta, phase, event, epoch=epoch, figure_dir=args.output_dir)
            del phase, event


def main(args):

    num_classes = 3

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(dataset)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.model == "phasenet_das":
        dataset = DASIterableDataset(
            # data_path="/net/kuafu/mnt/tank/data/EventData/Mammoth_north/data",
            # noise_path="/net/kuafu/mnt/tank/data/EventData/Mammoth_north/data",
            label_path=[
                "/net/kuafu/mnt/tank/data/EventData/Mammoth_north/picks_phasenet_filtered/",
                "/net/kuafu/mnt/tank/data/EventData/Mammoth_south/picks_phasenet_filtered/",
                "/net/kuafu/mnt/tank/data/EventData/Ridgecrest/picks_phasenet_filtered/",
                "/net/kuafu/mnt/tank/data/EventData/Ridgecrest_South/picks_phasenet_filtered/",
            ],
            format="h5",
            training=True,
        )
        train_sampler = None
        dataset_test = dataset = DASIterableDataset(
            # data_path="/net/kuafu/mnt/tank/data/EventData/Mammoth_south/data/",
            label_path=["/net/kuafu/mnt/tank/data/EventData/Mammoth_south/picks_phasenet_filtered/"],
            format="h5",
            stack_event=False,
            stack_noise=False,
            add_moveout=False,
            training=True,
        )
        test_sampler = None
    elif args.model == "autoencoder":
        dataset = AutoEncoderIterableDataset(
            # data_path = "/net/kuafu/mnt/tank/data/EventData/Ridgecrest/data",
            data_path=args.data_path,
            format="h5",
            training=True,
        )
        train_sampler = None
        dataset_test = dataset
        test_sampler = None
    elif args.model == "eqnet":
        dataset = SeismicNetworkIterableDataset(args.dataset)
        train_sampler = None
        dataset_test = dataset
        test_sampler = None
    elif args.model == "phasenet":
        dataset = SeismicTraceIterableDataset(args.dataset)
        train_sampler = None
        dataset_test = dataset
        test_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        # collate_fn=utils.collate_fn,
        collate_fn=None,
        # shuffle=True,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=None,
        # collate_fn=utils.collate_fn
    )

    model = eqnet.models.__dict__[args.model](backbone=args.backbone)
    logger.info("Model:\n{}".format(model))

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )  # , find_unused_parameters=True)
        model_without_ddp = model.module

    # params_to_optimize = [
    #     {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad], "lr": args.lr},
    #     {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    # ]
    # if args.aux_loss:
    #     params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10})

    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    # iters_per_epoch = 200
    # main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9
    # )
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iters_per_epoch * (args.epochs - args.lr_warmup_epochs)
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, args=args)
        print(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model, optimizer, data_loader, lr_scheduler, device, epoch, iters_per_epoch, args.print_freq, scaler, args
        )
        # confmat = evaluate(model, data_loader_test, device=device, args=args)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="./data/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="", type=str, help="dataset name")
    parser.add_argument("--model", default="phasenet_das", type=str, help="model name")
    parser.add_argument("--backbone", default="resnet50", type=str, help="model backbone")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16 * 2, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=1, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
