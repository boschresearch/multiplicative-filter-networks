import torch
import torchvision

from torch.optim import Adam
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, "../")

from mfn import FourierNet, GaborNet
from data import CameraDataset


def train(model, loader, opt, iterations=10000, device="cuda"):
    losses = []
    for it in range(1, iterations + 1):
        for x, y in loader:
            # load data, pass through model, get loss
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

def get_img_pred(model, loader, device="cuda"):
    # predict
    with torch.no_grad():
        x, _ = next(iter(loader))
        pred = (model(x.to(device))*0.5 + 0.5).clamp(0,1) # unnormalize
        pred = pred.detach().cpu().numpy()
    
    # reshape, convert to image, return
    side_length = int(np.sqrt(pred.shape[1]))
    pred = pred.reshape(side_length, side_length)

    return pred


def get_args(args):
    parser = ArgumentParser(description="Image representation task")

    parser.add_argument(
        "model", type=str, choices=["fourier", "gabor"], help="name of model"
    )
    parser.add_argument(
        "--save-dir", default=None, type=str, help="directory in which to save image & model"
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
        model = FourierNet(
            in_size=2,
            hidden_size=256,
            out_size=1,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
        ).to(device)

    elif args.model == "gabor":
        model = GaborNet(
            in_size=2,
            hidden_size=256,
            out_size=1,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
        ).to(device)

    opt = Adam(model.parameters(), lr=1e-2)

    # make dataset and loader
    camera_dataset = CameraDataset(side_length=256)
    loader = DataLoader(
        camera_dataset, batch_size=1, pin_memory=True
    )  # "batch" = entire image

    # train
    train(model, loader, opt, iterations=args.iterations)

    # save model state dict and represented image as np array, if path is supplied
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        predicted_img = get_img_pred(model, loader)
        np.save(os.path.join(args.save_dir, "represented_image.npy"), predicted_img)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"trained_model_{args.model}.pt"))

if __name__ == "__main__":
    main()
    
