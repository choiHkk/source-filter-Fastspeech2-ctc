import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import (
    get_model, 
    get_vocoder, 
    get_param_num
)
from utils.tools import (
    to_device, 
    log, 
    synth_one_sample, 
    plot_spectrogram_to_numpy, 
    plot_alignment_to_numpy
)
from model import FastSpeech2Loss
from data_utils import AudioTextDataset, AudioTextCollate, DataLoader
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs
    dataset = AudioTextDataset(
        preprocess_config['path']['training_files'], preprocess_config)
    
    batch_size = train_config["optimizer"]["batch_size"]
    collate_fn = AudioTextCollate()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=True
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config, train_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batch in loader:
            batch = to_device(batch, device)

            # Forward
            output = model(*(batch), step=step, gen=False)

            # Cal Loss
            losses = Loss(batch, output, step=step)
            total_loss = losses[0]

            # Backward
            total_loss = total_loss / grad_acc_step
            total_loss.backward()
            if step % grad_acc_step == 0:
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer.step_and_update_lr()
                optimizer.zero_grad()

            if step % log_step == 0:
                losses = [l.item() for l in losses]
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "TL: {:.4f}, ML: {:.4f}, PL: {:.4f}, DL: {:.4f}, CL: {:.4f}, BL: {:.4f}".format(
                    *losses
                )

                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")

                outer_bar.write(message1 + message2)
                
                for param_group in optimizer._optimizer.param_groups:
                    lr = param_group["lr"]
                
                scalar_dict = {
                    "loss/total_loss": losses[0],
                    "loss/mel_loss": losses[1],
                    "loss/pitch_loss": losses[2],
                    "loss/duration_loss": losses[3],
                    "loss/ctc_loss": losses[4],
                    "loss/bin_loss": losses[5],
                    "learning_rate": lr,
                }
                log(writer=train_logger,
                    global_step=step, 
                    scalars=scalar_dict)

            if step % val_step == 0:
                model.eval()
                message = evaluate(model, step, configs, val_logger, vocoder)
                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message + "\n")
                outer_bar.write(message)

                model.train()

            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                    },
                    os.path.join(
                        train_config["path"]["ckpt_path"],
                        "{}.pth.tar".format(step),
                    ),
                )

            if step == total_step:
                quit()
            step += 1
            outer_bar.update(1)

        inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)