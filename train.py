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

    train(model, train_loader, criterion, optimizer, config["EPOCHS"], device, logger,
          os.path.join(config["OUTPUT_DIR"], config["EXP_NAME"]))


def train(model, train_loader, criterion, optimizer, epochs, device, logger, output_dir) -> None:
    model.train()
    min_loss = float('inf')
    logger.info("Initiate complete! Start training...")
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)

        running_loss = torch.tensor(0.0).to(device)
        total_correct = torch.tensor(0.0).to(device)
        total_samples = torch.tensor(0.0).to(device)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            correct = (outputs.argmax(dim=1) == labels).sum().float()
            total_correct += correct
            total_samples += labels.size(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss

        torch.distributed.all_reduce(running_loss)
        torch.distributed.all_reduce(total_correct)
        torch.distributed.all_reduce(total_samples)

        avg_loss = running_loss.item() / total_samples.item()
        avg_accuracy = total_correct.item() / total_samples.item()

        if utils.is_main_process() and min_loss > avg_loss:
            min_loss = avg_loss
            torch.save(model.module.state_dict(), os.path.join(output_dir, "best.pth"))

        logger.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    logger.success("Finish training!")


if __name__ == '__main__':
    main()
