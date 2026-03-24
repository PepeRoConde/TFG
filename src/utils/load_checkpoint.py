import shutil
import torch


def save_checkpoint(state, is_best, args, file_name):
    checkpoint = args.weights_dir + "/checkpoint_" + file_name + ".pth.tar"
    best = args.weights_dir + "/" + file_name + ".pth.tar"

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)
