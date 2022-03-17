"""
Reads the configuration file and carries out the instructions to train the desired model.
"""

import argparse
import logging
import os
import torch
import warnings
import yaml


def main(args):
    
    config_path = os.path.join('./Config/',args.config_file)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = args.device
    data_file = config["data"]
    model_file = config["model"]
    augment_file = config["augment"]
    augment_strength = config["aug_strength"]
    eval_file = config["eval"]
    learn_rate = config["learning_rate"]
    epoch = config["epoch"]
    optimizer = config["optimizer"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    
    data_file_path = f"Data.{data_file}"
    _temp = __import__(name=data_file_path, fromlist=['Data_Load'])
    Data_Load = _temp.Data_Load
    Data_Load()
    
    Aug = []
    for i in range(len(augment_file)):
        augment_file_path = f"Augmentation.{augment_file[i]}"
        _temp = __import__(name=augment_file_path, fromlist=['Aug'])
        Aug.append(_temp.Aug)
    Aug[0]()
    Aug[1]()
    
    
    model_file_path = f"Model.{model_file}"
    _temp = __import__(name=model_file_path, fromlist=['Train'])
    Train = _temp.Train
    Train()
    
    eval_file_path = f"Evaluation.{eval_file}"
    _temp = __import__(name=eval_file_path, fromlist=['Eval'])
    Eval = _temp.Eval    
    Eval()
    
    
    return None






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFT6759 Experiments")

    parser.add_argument(
        "-c",
        "--config",
        type = str,
        dest="config_file",
        help="(string) name of the configuration file located in ./Config",
        default = "Test.yaml"
    )
    
    parser.add_argument(
        "-d"
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        dest="device",
        help="device to store tensors on (default: %(default)s).",
    )
    
    args = parser.parse_args()

    
    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is "
            "running on GPU (e.g. in the Notebook Settings in Google Colab). "
            'Forcing device="cpu".'
        )
        args.device = "cpu"

    else:
        warnings.warn(
            "You are about to run on CPU, and might run out of memory "
            "shortly. You can try setting batch_size=1 to reduce memory usage."
        )


    logs = main(args)





