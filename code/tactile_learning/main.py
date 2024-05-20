import os
from os.path import join
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import Tactile_Dataset
from model import TactilePoseNet
from utils import train_test_split


def train(train_videonames, test_videonames, args):

    mode = args.mode
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir

    torch.manual_seed(233)
    torch.cuda.manual_seed(233)
    np.random.seed(233)
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    batch_size = 32
    epoch = 50
    learning_rate = 1e-4

    print("exp setup: img_size = ({}, {}), traning span = [{}, {}], test span = {}, mode = {}".format(384, 288, args.train_span_L, args.train_span_R, args.test_span, mode))
    train_dataset = Tactile_Dataset(dataset_dir, train_videonames, device, img_size=(384, 288), min_span=args.train_span_L, max_span=args.train_span_R, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6, shuffle=True, drop_last=True)
    test_dataset = Tactile_Dataset(dataset_dir, test_videonames, device, img_size=(384, 288), min_span=args.test_span, max_span=args.test_span, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=6, shuffle=False, drop_last=False)

    # model = torch.nn.DataParallel(TactilePoseNet(), device_ids=[0,1,2,3]).cuda()
    model = TactilePoseNet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch_idx in range(epoch):
        print("traning epoch {} ...".format(epoch_idx))
        model.train()
        mean_training_loss = 0
        batch_cnt = 0
        for batch_idx, data in enumerate(train_dataloader):
            pred = model(data["rgb"])
            gt = torch.cat((data["R"], data["t"]), dim=1)
            if mode == "all":
                loss = loss_fn(pred, gt)
            elif mode == "R_only":
                loss = loss_fn(pred[:, :3], gt[:, :3])
            else:
                loss = loss_fn(pred[:, 3:], gt[:, 3:])
            mean_training_loss += loss.item()
            print("batch_idx = {}, loss = {}".format(batch_idx, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_cnt += 1
        
        mean_training_loss /= batch_cnt
        print("epoch: {}, mean training loss = {}".format(epoch_idx, mean_training_loss))

        if epoch_idx % 5 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), join(save_dir, "epoch_{}.pth".format(epoch_idx)))

            print("testing ...")
            model.eval()

            mean_test_loss = 0
            batch_cnt = 0
            mean_loss_R = 0
            mean_loss_t = 0

            for batch_idx, data in enumerate(test_dataloader):
                pred = model(data["rgb"])
                gt = torch.cat((data["R"], data["t"]), dim=1)
                if mode == "all":
                    loss = loss_fn(pred, gt)
                elif mode == "R_only":
                    loss = loss_fn(pred[:, :3], gt[:, :3])
                else:
                    loss = loss_fn(pred[:, 3:], gt[:, 3:])

                loss_R = ((pred[:, :3] - gt[:, :3]) * (pred[:, :3] - gt[:, :3])).sum(dim=1).sqrt().mean() / 5
                loss_t = ((pred[:, 3:] - gt[:, 3:]) * (pred[:, 3:] - gt[:, 3:])).sum(dim=1).sqrt().mean() / 100
                print("loss = {}, loss_R = {}(rad), loss_t = {}(m)".format(loss, loss_R, loss_t))
                print(list(pred[0].detach().cpu().numpy()), list(gt[0].detach().cpu().numpy()))
                mean_test_loss += loss.item()
                mean_loss_R += loss_R.item()
                mean_loss_t += loss_t.item()
                batch_cnt += 1
            
            print("mean testing loss = {}".format(mean_test_loss / batch_cnt))
            print("mean testing loss_R = {} (rad)".format(mean_loss_R / batch_cnt))
            print("mean testing loss_t = {} (m)".format(mean_loss_t / batch_cnt))


def inference():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="all")
    parser.add_argument('--dataset_dir', type=str, default="/nas/datasets/Visual_Tactile_Tracking_Dataset_Released")
    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--train_span_L', type=int, default=8)
    parser.add_argument('--train_span_R', type=int, default=15)
    parser.add_argument('--test_span', type=int, default=10)
    args = parser.parse_args()

    train_videonames, test_videonames = train_test_split()

    train(train_videonames, test_videonames, args)
