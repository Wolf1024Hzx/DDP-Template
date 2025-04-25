"""
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
from logger import Logger
from data import load_mnist
from models import TestModel


def main() -> None:
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)
    parser.add_argument("--config-path", type=str, default="./config/default.yaml")
    config = utils.load_config(parser.parse_args().config_path)
    logger = Logger(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)

    _, val_loader = load_mnist(batch_size=config['BATCH_SIZE'])
    model = TestModel().to(device)

    if dist.get_rank() == 0 and config["INFERENCE_MODEL_PATH"] is not None:
        model.load_state_dict(torch.load(config["INFERENCE_MODEL_PATH"], map_location=device))
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss().to(device)

    evaluate(model, val_loader, criterion, device, logger)


def evaluate(model, val_loader, criterion, device, logger) -> None:
    model.eval()

    running_loss = torch.tensor(0.0).to(device)
    total_correct = torch.tensor(0.0).to(device)
    total_samples = torch.tensor(0.0).to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            correct = (outputs.argmax(dim=1) == labels).sum().float()
            total_correct += correct
            total_samples += labels.size(0)
            running_loss += loss.item()

    torch.distributed.all_reduce(total_correct)
    torch.distributed.all_reduce(total_samples)
    torch.distributed.all_reduce(running_loss)

    avg_accuracy = total_correct.item() / total_samples.item()
    avg_loss = running_loss.item() / total_samples.item()
    logger.info(f"Evaluation Results - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")


def count_accuracy(outputs, labels):
    outputs = outputs.argmax(axis=1)
    cmp = outputs.type(labels.dtype) == labels
    return float(cmp.type(labels.dtype).sum())


if __name__ == '__main__':
    main()
