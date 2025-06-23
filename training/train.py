# Warsaw University of Technology
# Train MinkLoc model

import argparse
import torch
import sys
import os
from training.trainer import do_train
from misc.utils import TrainingParams


script_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MinkLoc3Dv2 model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--data_path', type=str, required=False, help='Override dataset folder path')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))

    params = TrainingParams(args.config, args.model_config, debug=args.debug, dataset_folder_override=args.data_path)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    do_train(params)
