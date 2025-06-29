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
import pickle
import tqdm
import torch
import numpy as np
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree
from sklearn.metrics import roc_curve, auc

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader

import wandb

PERCENTAGE_TO_REMOVE_STEP = 0.1
NUM_NEIGHBORS = 25

def evaluate_der(model, device, params: TrainingParams, log: bool = True, show_progress: bool = False):
    """
    Funzione principale per avviare la valutazione su tutti i dataset PointNetVLAD.
    """
    params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    params_dict.update(model_params_dict)

    if log:
        if "WANDB_API_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        wandb.init(
            project='MinkLoc3D-EvD-Eval',
            config=params_dict,
            mode=os.getenv("WANDB_MODE", "online"),
            dir=os.getenv("AZUREML_OUTPUT_DIR", "./wandb_logs"))

    eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
                           'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']
    eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
                        'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    all_stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        location_name = database_file.split('_')[0]
        print(f"Evaluating on {location_name}...")
        with open(os.path.join(params.dataset_folder, database_file), 'rb') as f:
            database_sets = pickle.load(f)
        with open(os.path.join(params.dataset_folder, query_file), 'rb') as f:
            query_sets = pickle.load(f)

        dataset_stats = evaluate_dataset(model, device, params, database_sets, query_sets, location_name,
                                         show_progress=show_progress)
        all_stats.update(dataset_stats)

    log_uncertainty_eval_to_wandb(all_stats)


    return all_stats

def calculate_auroc_fdr(db_embeddings, query_embeddings, query_uncertainties, query_sets, database_sets):
    """
    Calcola AUROC e Failure Detection Rate (FDR).
    """
    all_top1_errors = []
    all_query_uncertainties = []

    for i in range(len(query_sets)):
        for j in range(len(database_sets)):
            if i == j:
                continue

            current_db_emb = db_embeddings[j]
            current_query_emb = query_embeddings[i]
            current_query_unc = query_uncertainties[i]
            total_uncertainty_scores = np.sum(current_query_unc, axis=1)
            
            # Costruisce il ground truth per la coppia corrente
            current_gt = {idx: query_sets[i][idx].get(j, []) for idx in range(len(query_sets[i]))}
            
            database_kdtree = KDTree(current_db_emb)
            
            for q_idx in range(len(current_query_emb)):
                true_positives = current_gt.get(q_idx, [])
                if not true_positives:
                    continue

                # Cerca il vicino più prossimo (top-1)
                retrieved_indices = database_kdtree.query(np.array([current_query_emb[q_idx]]), k=1, return_distance=False)
                retrieved_idx = retrieved_indices[0, 0]

                is_error = 1 if retrieved_idx not in true_positives else 0
                all_top1_errors.append(is_error)
                all_query_uncertainties.append(total_uncertainty_scores[q_idx])

    if not all_top1_errors:
        return None

    # Calcola AUROC
    fpr, tpr, _ = roc_curve(all_top1_errors, all_query_uncertainties)
    auroc_score = auc(fpr, tpr)
    
    # Calcola Failure Detection Rate @ 80% Rejection Rate
    # (Tasso di rilevamento dei fallimenti al tasso di rigetto dell'80%)
    try:
        threshold = np.percentile(all_query_uncertainties, 80)
        rejected_mask = np.array(all_query_uncertainties) >= threshold
        errors_mask = np.array(all_top1_errors) == 1
        
        correct_rejections = np.sum(rejected_mask & errors_mask)
        total_errors = np.sum(errors_mask)
        
        fdr_at_80 = correct_rejections / total_errors if total_errors > 0 else 0.0
    except IndexError:
        fdr_at_80 = 0.0

    return {
        'auroc': auroc_score,
        'fdr_at_80': fdr_at_80,
        'fpr': fpr,
        'tpr': tpr
    }


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets, location_name: str,
                     show_progress: bool = False):
    """
    Valuta un singolo dataset, calcolando sia le metriche di recall che AUROC/FDR.
    """
    db_embeddings, _ = get_embeddings_and_uncertainties(model, database_sets, device, params, f'Computing database embeddings for {location_name}', show_progress=show_progress)
    query_embeddings, query_uncertainties = get_embeddings_and_uncertainties(model, query_sets, device, params, f'Computing query embeddings for {location_name}', show_progress=show_progress)

    dataset_stats = {}

    # --- Calcolo AUROC e FDR ---
    print(f"Calculating AUROC/FDR for {location_name}...")
    auroc_stats = calculate_auroc_fdr(db_embeddings, query_embeddings, query_uncertainties, query_sets, database_sets)
    if auroc_stats:
        dataset_stats[f"{location_name}_auroc"] = auroc_stats

    # --- Calcolo Recall vs. % Dati Rimossi ---
    print(f"Calculating Recall vs. data removal for {location_name}...")
    num_removal_steps = int(1.0 / PERCENTAGE_TO_REMOVE_STEP) + 1

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
                original_gt = {idx: query_sets[i][idx].get(j, []) for idx in range(len(query_sets[i]))}
                filtered_ground_truth = {new_idx: original_gt[old_idx] for new_idx, old_idx in enumerate(indices_to_keep)}

                pair_recall, pair_opr = get_recall(db_embeddings[j], filtered_query_embeddings, filtered_ground_truth)

                recall_sum += pair_recall
                one_percent_recall_sum.append(pair_opr)
                pair_count += 1
        
        if pair_count > 0:
            avg_recall = recall_sum / pair_count
            avg_one_percent_recall = np.mean(one_percent_recall_sum)
        else:
            avg_recall = np.zeros(NUM_NEIGHBORS)
            avg_one_percent_recall = 0.0

        stats_key = f"{location_name}_recall_{100 - int(percentage_to_remove * 100)}%"
        dataset_stats[stats_key] = {'ave_one_percent_recall': avg_one_percent_recall, 'ave_recall': avg_recall}

    return dataset_stats


def get_embeddings_and_uncertainties(model, sets, device, params, description, show_progress=False):
    """
    Calcola gli embedding e i valori di incertezza per un dato set.
    """
    all_embeddings = []
    all_uncertainties = []
    for set_data in tqdm.tqdm(sets, desc=description, disable=not show_progress):
        embeddings = np.zeros((len(set_data), params.model_params.output_dim))
        uncertainties = np.zeros((len(set_data), params.model_params.output_dim))
        for i, elem_ndx in enumerate(set_data):
            pred = compute_embedding(model, set_data[elem_ndx]["query"], device, params)
            gamma = pred['gamma'].detach().cpu().numpy().squeeze()
            nu = pred['nu'].detach().cpu().numpy().squeeze()
            alpha = pred['alpha'].detach().cpu().numpy().squeeze()
            beta = pred['beta'].detach().cpu().numpy().squeeze()
            
            embeddings[i] = gamma
            # Calcolo dell'incertezza come varianza della distribuzione Normal-Inverse-Gamma
            uncertainties[i] = beta / (nu * (alpha - 1.0))
        all_embeddings.append(embeddings)
        all_uncertainties.append(uncertainties)
    return all_embeddings, all_uncertainties


def compute_embedding(model, pc_path, device, params: TrainingParams):
    """
    Calcola l'embedding per una singola nuvola di punti.
    """
    pc_loader = PNVPointCloudLoader()
    pc = torch.tensor(pc_loader(os.path.join(params.dataset_folder, pc_path)))
    
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        model.eval()
        batch = {
            'coords': ME.utils.batched_coordinates([coords]).to(device),
            'features': torch.ones((len(coords), 1), dtype=torch.float32).to(device)
        }
        prediction = model(batch)
    return prediction


def get_indices_to_keep(uncertainties, percentage_to_remove):
    """
    Restituisce gli indici degli elementi da mantenere, scartando quelli con l'incertezza più alta.
    """
    if percentage_to_remove == 0:
        return np.arange(len(uncertainties))
    # L'incertezza totale è la somma delle incertezze per dimensione
    total_uncertainty_score = np.sum(uncertainties, axis=1)
    sorted_indices = np.argsort(total_uncertainty_score)
    num_to_keep = len(uncertainties) - int(len(uncertainties) * percentage_to_remove)
    return sorted_indices[:num_to_keep]


def get_recall(database_embeddings, query_embeddings, query_ground_truth):
    """
    Calcola il recall@N e il recall@1%.
    """
    database_kdtree = KDTree(database_embeddings)
    recall = np.zeros(NUM_NEIGHBORS)
    one_percent_retrieved = 0
    threshold = max(1, int(round(len(database_embeddings) / 100.0)))
    num_evaluated = 0

    for i in range(len(query_embeddings)):
        true_positives = query_ground_truth.get(i, [])
        if not true_positives:
            continue
        
        num_evaluated += 1
        # Esegue la query per trovare i k vicini più prossimi
        indices = database_kdtree.query(np.array([query_embeddings[i]]), k=NUM_NEIGHBORS, return_distance=False)
        
        # Controlla se c'è una corrispondenza nei primi k risultati
        if len(set(indices[0]).intersection(set(true_positives))) > 0:
            for j in range(NUM_NEIGHBORS):
                if indices[0, j] in true_positives:
                    recall[j:] += 1
                    break
        
        # Controlla se c'è una corrispondenza nel top 1%
        if len(set(indices[0, :threshold]).intersection(set(true_positives))) > 0:
            one_percent_retrieved += 1

    if num_evaluated == 0:
        return np.zeros(NUM_NEIGHBORS), 0.0

    return (recall / num_evaluated) * 100, (one_percent_retrieved / num_evaluated) * 100

def print_eval_stats(stats):
    """
    Stampa le statistiche di valutazione sulla console.
    """
    # Filtra prima le statistiche per raggrupparle per dataset
    recall_stats = defaultdict(dict)
    auroc_stats = {}
    
    for key, data in stats.items():
        if "_auroc" in key:
            dataset_name = key.replace("_auroc", "")
            auroc_stats[dataset_name] = data
        elif "_recall_" in key:
            match = re.match(r"(.+?)_recall_(\d+)%", key)
            if match:
                dataset, keep_pct = match.groups()
                # Considera solo il caso con 100% dei dati per la stampa
                if int(keep_pct) == 100:
                    recall_stats[dataset] = data
    
    for dataset_name in sorted(recall_stats.keys()):
        print(f'\n--- Dataset: {dataset_name} ---')
        if dataset_name in recall_stats:
            r_stats = recall_stats[dataset_name]
            t = 'Avg. top 1% recall: {:.2f}   Avg. recall @1: {:.2f}'
            print(t.format(r_stats['ave_one_percent_recall'], r_stats['ave_recall'][0]))
        if dataset_name in auroc_stats:
            a_stats = auroc_stats[dataset_name]
            t = 'AUROC: {:.4f}   Failure Detection Rate @80% Reject: {:.2f}%'
            print(t.format(a_stats['auroc'], a_stats['fdr_at_80'] * 100))


def pnv_write_eval_stats(file_name, prefix, stats):
    """
    Scrive un riepilogo delle statistiche su un file di testo.
    """
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    
    # Considera solo le chiavi di recall con il 100% dei dati
    recall_keys = [k for k in stats if "_recall_100%" in k]

    with open(file_name, "a") as f:
        for ds_key in sorted(recall_keys):
            ds_stats = stats[ds_key]
            ave_1p_recall = ds_stats['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = ds_stats['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        if ave_1p_recall_l:
            mean_1p_recall = np.mean(ave_1p_recall_l)
            mean_recall = np.mean(ave_recall_l)
            s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
            f.write(s)


def log_uncertainty_eval_to_wandb(stats, prefix="uncertainty_eval/"):
    """
    Registra le statistiche di valutazione su Weights & Biases.
    """
    if not wandb.run:
        print("WandB run not initialized. Skipping logging.")
        return

    # Gestisce le metriche AUROC/FDR
    auroc_stats = {k.replace("_auroc", ""): v for k, v in stats.items() if "_auroc" in k}
    for dataset_name, data in auroc_stats.items():
        roc_table = wandb.Table(data=list(zip(data['fpr'], data['tpr'])), columns=["FPR", "TPR"])
        wandb.log({
            f"{prefix}{dataset_name}/AUROC": data['auroc'],
            f"{prefix}{dataset_name}/Failure_Detection_Rate_80": data['fdr_at_80'],
            f"{prefix}{dataset_name}/Uncertainty_ROC_Curve": wandb.plot.line(roc_table, "FPR", "TPR", title=f"Uncertainty ROC Curve - {dataset_name}")
        })


    # Gestisce le metriche di recall
    recall_stats = defaultdict(dict)
    for key, data in stats.items():
        match = re.match(r"(.+?)_recall_(\d+)%", key)
        if match:
            dataset, keep_pct = match.groups()
            recall_stats[dataset][int(keep_pct)] = data

    for dataset_name, pct_data in recall_stats.items():
        sorted_pcts = sorted(pct_data.keys(), reverse=True)

        columns = ["% Kept", "Avg top 1% Recall"] + [f"Recall@{i+1}" for i in range(NUM_NEIGHBORS)]
        summary_table = wandb.Table(columns=columns)

        for pct in sorted_pcts:
            data = pct_data[pct]
            row = [pct, data['ave_one_percent_recall']] + list(data['ave_recall'])
            summary_table.add_data(*row)

        wandb.log({
            f"{prefix}{dataset_name}/Recall_Summary_Table": summary_table
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate uncertainty-aware model on PointNetVLAD datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the dataset folder (optional)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--show_progress', action='store_true', help='Show tqdm progress bars')


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

    stats = evaluate_der(model, device, params, show_progress=False, log=True)
    print_eval_stats(stats)

    # Opzionale: salva i risultati su file
    # model_params_name = os.path.basename(params.model_params.model_params_path)
    # config_name = os.path.basename(params.params_path)
    # model_name = os.path.splitext(os.path.basename(args.weights))[0]
    # prefix = f"{model_params_name}, {config_name}, {model_name}"
    # pnv_write_eval_stats("./outputs/pnv_experiment_results.txt", prefix, stats)
    print("\nEvaluation complete.")