import torch
import torchvision

from torch.optim import Adam
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import numpy as np
import os
import sys

sys.path.insert(0, "../")

from mfn import FourierNet, GaborNet
from data import VideoDataset


def train(model, video_data, opt, iterations=10000, batch_size=298844, device="cuda"):
    losses = []
    for it in range(1, iterations+1):
        # get data idx and load (much faster than default dataloader)
        idx = torch.randint(0, len(video_data), (batch_size,))
        x, y = video_data.coords[idx], video_data.video[idx]

        # pass through model, get loss
        preds = model(x.to(device))
        loss = ((preds - y.to(device)) ** 2).mean()

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if not it % 20:
            print(
                f"iter: {it:>5d}, loss: {loss.item():2.3e}, PSNR: {-10*np.log10(loss.item()):.2f}"
            )

    return losses


def get_args(args):
    parser = ArgumentParser(description="Image representation task")

    parser.add_argument(
        "model", type=str, choices=["fourier", "gabor"], help="name of model"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="path to video; expects numpy array of shape (timesteps, x size, y size)",
    )
    parser.add_argument(
        "--save-dir", default=None, type=str, help="path to save trained model"
    )
    parser.add_argument(
        "--iterations", default=10000, type=int, help="Number of training iterations"
    )

    return parser.parse_args(args)


def main():
    device = "cuda"
    args = get_args(sys.argv[1:])

    # make model and optimizer
    if args.model == "fourier":
        model = GaborNet(
            in_size=3,  # time, x coord, y coord
            hidden_size=1024,
            out_size=3,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
        ).to(device)
    elif args.model == "gabor":
        model = FourierNet(
            in_size=3,
            hidden_size=1024,
            out_size=3,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
        ).to(device)

    model = torch.nn.DataParallel(model)
    opt = Adam(model.parameters(), lr=1e-3)

    # make dataset
    video_data = VideoDataset(args.video_path)

    # train
    train(model, video_data, opt, iterations=args.iterations)

    # save model dict, if save dir is supplied
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, f"trained_model_{args.model}.pt"))


if __name__ == "__main__":
    main()
