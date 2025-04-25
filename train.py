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

    train_loader, _ = load_mnist(batch_size=config['BATCH_SIZE'])
    model = TestModel().to(device)

    if dist.get_rank() == 0 and config["CHECKPOINT_PATH"] is not None:
        model.load_state_dict(torch.load(config["CHECKPOINT_PATH"], map_location=device))
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=config['LR'])
    criterion = nn.CrossEntropyLoss().to(device)

    train(model, train_loader, criterion, optimizer, config["EPOCHS"], device, logger, os.path.join(config["OUTPUT_DIR"], config["EXP_NAME"]))


def train(model, train_loader, criterion, optimizer, epochs, device, logger, output_dir) -> None:
    model.train()
    min_loss = float('inf')
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        running_loss = 0.0
        train_loader.sampler.set_epoch(epoch)
        total_accuracy = 0.0
        # finish_len = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            acc = count_accuracy(outputs, labels) / len(labels)
            total_accuracy += acc
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # finish_len += 1
            # logger.info(
            #     f'Epoch {epoch + 1}[{finish_len}/{len(train_loader)}] Loss: {running_loss / len(train_loader):.4f} Accuracy: {total_accuracy / len(train_loader):.4f}')

        loss_values.append(running_loss / len(train_loader))

        if utils.is_main_process() and min_loss > running_loss:
            min_loss = running_loss
            torch.save(model.module.state_dict(), os.path.join(output_dir, "best.pth"))

        total_accuracy = total_accuracy / len(train_loader)
        accuracy_values.append(total_accuracy)
        # logger.info(f"Finish epoch {epoch + 1}")
        logger.info(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {total_accuracy:.4f}")
    logger.success("Finish training!")


def count_accuracy(outputs, labels):
    outputs = outputs.argmax(axis=1)
    cmp = outputs.type(labels.dtype) == labels
    return float(cmp.type(labels.dtype).sum())


if __name__ == '__main__':
    main()
