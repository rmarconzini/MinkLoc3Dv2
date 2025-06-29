# pnv_evaluate_uncertainty.py

from collections import defaultdict
import re
import sys
import os

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import os
import pickle
import tqdm
import torch
import numpy as np
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader

import wandb


if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

PERCENTAGE_TO_REMOVE_STEP = 0.1
NUM_NEIGHBORS = 25

def evaluate_der(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False):


    params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    params_dict.update(model_params_dict)

    wandb.init(
        project='MinkLoc3D-EvD-Eval',
        config=params_dict,
        mode=os.getenv("WANDB_MODE", "offline"),
        dir=os.getenv("AZUREML_OUTPUT_DIR", "./wandb_logs"))

    eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
                           'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']
    eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
                        'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    all_stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        location_name = database_file.split('_')[0]
        with open(os.path.join(params.dataset_folder, database_file), 'rb') as f:
            database_sets = pickle.load(f)
        with open(os.path.join(params.dataset_folder, query_file), 'rb') as f:
            query_sets = pickle.load(f)

        dataset_stats = evaluate_dataset(model, device, params, database_sets, query_sets, location_name,
                                         log=log, show_progress=show_progress)
        all_stats.update(dataset_stats)

    return all_stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets, location_name: str,
                     log: bool = False, show_progress: bool = False):
    db_embeddings, _ = get_embeddings_and_uncertainties(model, database_sets, device, params, 'Computing database embeddings', show_progress=show_progress)
    query_embeddings, query_uncertainties = get_embeddings_and_uncertainties(model, query_sets, device, params, 'Computing query embeddings', show_progress=show_progress)

    dataset_stats = {}
    num_removal_steps = int(1 / PERCENTAGE_TO_REMOVE_STEP)

    for k in range(num_removal_steps):
        percentage_to_remove = k * PERCENTAGE_TO_REMOVE_STEP
        recall_sum = np.zeros(NUM_NEIGHBORS)
        one_percent_recall_sum = []
        pair_count = 0

        for i in range(len(query_sets)):
            for j in range(len(database_sets)):
                if i == j:
                    continue

                indices_to_keep = get_indices_to_keep(query_uncertainties[i], percentage_to_remove)
                filtered_query_embeddings = query_embeddings[i][indices_to_keep]

                # Filtra il ground truth in base agli indici mantenuti
                original_query_set = query_sets[i]
                filtered_ground_truth = {new_idx: original_query_set[old_idx][j] for new_idx, old_idx in enumerate(indices_to_keep)}

                pair_recall, pair_opr = get_recall(db_embeddings[j], filtered_query_embeddings, filtered_ground_truth)

                recall_sum += pair_recall
                one_percent_recall_sum.append(pair_opr)
                pair_count += 1
        
        avg_recall = recall_sum / pair_count
        avg_one_percent_recall = np.mean(one_percent_recall_sum)
        
        # **CORREZIONE**: Usa location_name per creare la chiave
        stats_key = f"{location_name}_{100 - int(percentage_to_remove * 100)}%"
        dataset_stats[stats_key] = {'ave_one_percent_recall': avg_one_percent_recall, 'ave_recall': avg_recall}

    return dataset_stats


def get_embeddings_and_uncertainties(model, sets, device, params, description, show_progress=False):

    all_embeddings = []
    all_uncertainties = []
    for set_data in tqdm.tqdm(sets, desc=description, disable=not show_progress):
        embeddings = np.zeros((len(set_data), params.model_params.output_dim))
        uncertainties = np.zeros((len(set_data), params.model_params.output_dim))
        for i, elem_ndx in enumerate(set_data):
            pred = compute_embedding(model, set_data[elem_ndx]["query"], device, params)
            gamma = pred['gamma'].detach().cpu().numpy()
            nu = pred['nu'].detach().cpu().numpy()
            alpha = pred['alpha'].detach().cpu().numpy()
            beta = pred['beta'].detach().cpu().numpy()
            
            embeddings[i] = gamma
            uncertainties[i] = beta / (nu * (alpha - 1.0))
        all_embeddings.append(embeddings)
        all_uncertainties.append(uncertainties)
    return all_embeddings, all_uncertainties


def compute_embedding(model, pc_path, device, params: TrainingParams):

    pc_loader = PNVPointCloudLoader()
    pc = torch.tensor(pc_loader(os.path.join(params.dataset_folder, pc_path)))
    
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        batch = {
            'coords': ME.utils.batched_coordinates([coords]).to(device),
            'features': torch.ones((len(coords), 1), dtype=torch.float32).to(device)
        }
        prediction = model(batch)
    return prediction


def get_indices_to_keep(uncertainties, percentage_to_remove):

    if percentage_to_remove == 0:
        return np.arange(len(uncertainties))
    total_uncertainty_score = np.sum(uncertainties, axis=1)
    sorted_indices = np.argsort(total_uncertainty_score)
    num_to_keep = len(uncertainties) - int(len(uncertainties) * percentage_to_remove)
    return sorted_indices[:num_to_keep]


def get_recall(database_embeddings, query_embeddings, query_ground_truth):

    database_kdtree = KDTree(database_embeddings)
    recall = np.zeros(NUM_NEIGHBORS)
    one_percent_retrieved = 0
    threshold = max(1, int(round(len(database_embeddings) / 100.0)))
    num_evaluated = 0

    for i in range(len(query_embeddings)):
        true_positives = query_ground_truth[i]
        if len(true_positives) == 0:
            continue
        
        num_evaluated += 1
        indices = database_kdtree.query(np.array([query_embeddings[i]]), k=NUM_NEIGHBORS, return_distance=False)
        
        if len(set(indices[0]).intersection(set(true_positives))) > 0:
            for j in range(NUM_NEIGHBORS):
                if indices[0, j] in true_positives:
                    recall[j:] += 1
                    break
        
        if len(set(indices[0, :threshold]).intersection(set(true_positives))) > 0:
            one_percent_retrieved += 1

    if num_evaluated == 0:
        return np.zeros(NUM_NEIGHBORS), 0.0

    return (recall / num_evaluated) * 100, (one_percent_retrieved / num_evaluated) * 100

def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall']))
        print(stats[database_name]['ave_recall'])

def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


def log_uncertainty_eval_to_wandb(stats, prefix="uncertainty_eval/"):

    from collections import defaultdict
    import re
    import wandb

    grouped_stats = defaultdict(dict)
    for key, data in stats.items():
        match = re.match(r"(.+?)_(\d+)%", key)
        if match:
            dataset, keep_pct = match.groups()
            grouped_stats[dataset][int(keep_pct)] = data

    for dataset_name, pct_data in grouped_stats.items():
        sorted_pcts = sorted(pct_data.keys(), reverse=True)

        columns = ["% Kept", "Avg top 1% Recall"] + [f"Recall@{i}" for i in range(1, NUM_NEIGHBORS + 1)]
        summary_table = wandb.Table(columns=columns)

        for pct in sorted_pcts:
            data = pct_data[pct]
            row = [pct, data['ave_one_percent_recall']] + list(data['ave_recall'])
            summary_table.add_data(*row)

        wandb.log({
            f"{prefix}{dataset_name}/Summary_Table": summary_table
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate uncertainty-aware model on PointNetVLAD datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model weights')
    # Optional: data path to override the one in the config file
    parser.add_argument('--data_path', type=str, default=None, help='Path to the dataset folder (optional)')

    args = parser.parse_args()
    print(f"Config path: {args.config}")
    print(f"Model config path: {args.model_config}")
    print(f"Weights: {args.weights}\n")

    params = TrainingParams(args.config, args.model_config, dataset_folder_override=args.data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = model_factory(params.model_params)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    stats = evaluate_der(model, device, params, show_progress=False)
    print_eval_stats(stats)
    log_uncertainty_eval_to_wandb(stats)

    # model_params_name = os.path.split(params.model_params.model_params_path)[1]
    # config_name = os.path.split(params.params_path)[1]
    # model_name = os.path.split(args.weights)[1]
    # model_name = os.path.splitext(model_name)[0]    
    # prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

    # pnv_write_eval_stats("./outputs/pnv_experiment_results_DER_based_triplet_loss.txt", prefix, stats)
