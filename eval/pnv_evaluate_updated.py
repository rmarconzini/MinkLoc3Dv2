# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

import sys
import os

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.neighbors import KDTree
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import tqdm

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader

import wandb


def evaluate(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False):
    """
    Funzione principale che avvia la valutazione su tutti i dataset.
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

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        location_name = database_file.split('_')[0]
        print(f"Evaluating on {location_name}...")
        
        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        dataset_stats = evaluate_dataset(model, device, params, database_sets, query_sets)
        stats[location_name] = dataset_stats

        log_standard_eval_to_wandb(stats)

    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets):
    """
    Esegue la valutazione su un singolo dataset, calcolando recall, mAP e curva PR.
    """
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []
    
    # Liste per raccogliere i dati per mAP e PR-curve
    all_y_true = []
    all_y_scores = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    # Calcolo degli embedding per il database e le query
    for set_data in tqdm.tqdm(database_sets, desc='Computing database embeddings'):
        database_embeddings.append(get_latent_vectors(model, set_data, device, params))

    for set_data in tqdm.tqdm(query_sets, desc='Computing query embeddings'):
        query_embeddings.append(get_latent_vectors(model, set_data, device, params))

    # Calcolo delle metriche per ogni coppia di run
    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
            pair_recall, pair_opr, pair_y_true, pair_y_scores = get_recall(i, j, database_embeddings, query_embeddings, query_sets, database_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            all_y_true.extend(pair_y_true)
            all_y_scores.extend(pair_y_scores)
    
    # Calcolo delle metriche aggregate
    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)

    # Calcolo di mAP e curva PR
    map_score = average_precision_score(all_y_true, all_y_scores)
    precision, pr_recall, _ = precision_recall_curve(all_y_true, all_y_scores)

    stats = {
        'ave_one_percent_recall': ave_one_percent_recall,
        'ave_recall': ave_recall,
        'mAP': map_score,
        'pr_precision': precision,
        'pr_recall': pr_recall
    }
    return stats


def get_latent_vectors(model, set_data, device, params: TrainingParams):
    """
    Calcola gli embedding per un dato set di nuvole di punti.
    """
    pc_loader = PNVPointCloudLoader()
    model.eval()
    embeddings = None
    for i, elem_ndx in enumerate(set_data):
        pc_file_path = os.path.join(params.dataset_folder, set_data[elem_ndx]["query"])
        pc = pc_loader(pc_file_path)
        pc = torch.tensor(pc)

        embedding = compute_embedding(model, pc, device, params)
        if embeddings is None:
            # Inizializza l'array di embedding con la forma corretta
            embeddings = np.zeros((len(set_data), embedding.shape[0]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings


def compute_embedding(model, pc, device, params: TrainingParams):
    """
    Calcola l'embedding per una singola nuvola di punti.
    """
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        y = model(batch)
        # Assumendo che il modello restituisca un dizionario con 'global'
        embedding = y['global'].detach().cpu().numpy().squeeze()

    return embedding


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    """
    Calcola il recall e raccoglie i dati per mAP/PR per una coppia di run (m, n).
    """
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    
    y_true_for_map = []
    y_scores_for_map = []

    num_evaluated = 0
    for i in range(len(queries_output)):
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        # Raccoglie dati per mAP basati sulla previsione top-1
        is_correct_top1 = 1 if indices[0, 0] in true_neighbors else 0
        # Il punteggio è la distanza negativa (più alto è, meglio è)
        score_top1 = -distances[0, 0]
        y_true_for_map.append(is_correct_top1)
        y_scores_for_map.append(score_top1)

        # Calcolo del Recall@N
        if any(idx in true_neighbors for idx in indices[0]):
            first_match_idx = next(j for j, idx in enumerate(indices[0]) if idx in true_neighbors)
            recall[first_match_idx:] = [val + 1 for val in recall[first_match_idx:]]

        # Calcolo del Recall@1%
        if any(idx in true_neighbors for idx in indices[0, :threshold]):
            one_percent_retrieved += 1

    if num_evaluated == 0:
        return recall, 0.0, [], []

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.array(recall) / float(num_evaluated)) * 100
    
    return recall, one_percent_recall, y_true_for_map, y_scores_for_map


def print_eval_stats(stats):
    """
    Stampa a console le statistiche di valutazione.
    """
    for database_name in stats:
        print(f'\n--- Dataset: {database_name} ---')
        t = 'Avg. top 1% recall: {top1_recall:.2f}   |   mAP: {map:.4f}   |   Avg. recall @1: {r1:.2f}'
        print(t.format(top1_recall=stats[database_name]['ave_one_percent_recall'],
                       map=stats[database_name]['mAP'],
                       r1=stats[database_name]['ave_recall'][0]))


def pnv_write_eval_stats(file_name, prefix, stats):
    """
    Scrive le statistiche di valutazione su un file.
    """
    s = prefix
    with open(file_name, "a") as f:
        all_aps = [stats[ds]['ave_one_percent_recall'] for ds in stats]
        all_recalls = [stats[ds]['ave_recall'][0] for ds in stats]
        all_maps = [stats[ds]['mAP'] for ds in stats]

        for ds in stats:
            s += ", {:0.2f}, {:0.2f}, {:0.4f}".format(stats[ds]['ave_one_percent_recall'], stats[ds]['ave_recall'][0], stats[ds]['mAP'])

        s += ", {:0.2f}, {:0.2f}, {:0.4f}\n".format(np.mean(all_aps), np.mean(all_recalls), np.mean(all_maps))
        f.write(s)


def log_standard_eval_to_wandb(stats, prefix="standard_eval/"):
    """
    Logga i risultati della valutazione standard su W&B, incluse le nuove metriche.
    """
    all_datasets_data = {
        "Avg top 1% Recall": [],
        "mAP": [],
        "Dataset": []
    }

    for dataset_name, dataset_stats in stats.items():
        # Log curva Recall@N
        recalls = dataset_stats['ave_recall']
        n_values = list(range(1, len(recalls) + 1))
        recall_table = wandb.Table(data=[[n, r] for n, r in zip(n_values, recalls)], columns=["N", "Recall"])
        wandb.log({f"{prefix}{dataset_name}/Recall_Curve": wandb.plot.line(recall_table, "N", "Recall", title=f"{dataset_name}: Recall@N")})

        # Log curva Precision-Recall
        pr_precision = dataset_stats['pr_precision']
        pr_recall = dataset_stats['pr_recall']
        pr_table = wandb.Table(data=[[r, p] for r, p in zip(pr_recall, pr_precision)], columns=["Recall", "Precision"])
        wandb.log({f"{prefix}{dataset_name}/Precision_Recall_Curve": wandb.plot.line(pr_table, "Recall", "Precision", title=f"{dataset_name}: Precision-Recall Curve")})
        
        # Log mAP scalare per il dataset
        wandb.log({f"{prefix}{dataset_name}/mAP": dataset_stats['mAP']})

        # Raccoglie i dati per i grafici a barre comparativi
        all_datasets_data["Dataset"].append(dataset_name)
        all_datasets_data["Avg top 1% Recall"].append(dataset_stats['ave_one_percent_recall'])
        all_datasets_data["mAP"].append(dataset_stats['mAP'])

    # Crea e logga i grafici a barre comparativi
    bar_table_recall = wandb.Table(data=[[d, r] for d, r in zip(all_datasets_data["Dataset"], all_datasets_data["Avg top 1% Recall"])], columns=["Dataset", "Avg top 1% Recall"])
    wandb.log({f"{prefix}Avg_Top_1_Percent_Recall_Comparison": wandb.plot.bar(bar_table_recall, "Dataset", "Avg top 1% Recall", title="Avg top 1% Recall Comparison")})

    bar_table_map = wandb.Table(data=[[d, m] for d, m in zip(all_datasets_data["Dataset"], all_datasets_data["mAP"])], columns=["Dataset", "mAP"])
    wandb.log({f"{prefix}mAP_Comparison": wandb.plot.bar(bar_table_map, "Dataset", "mAP", title="mAP Comparison")})

    # Crea e logga la tabella riassuntiva
    columns = ["Dataset", "Avg top 1% Recall", "mAP"] + [f"Recall@{i}" for i in range(1, 26)]
    summary_table = wandb.Table(columns=columns)
    for dataset_name, dataset_stats in stats.items():
        row = [dataset_name, dataset_stats['ave_one_percent_recall'], dataset_stats['mAP']] + list(dataset_stats['ave_recall'])
        summary_table.add_data(*row)
    wandb.log({f"{prefix}Summary_Table": summary_table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Trained model weights')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the dataset folder (optional)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')

    args = parser.parse_args()
    print(f"Config path: {args.config}")
    print(f"Model config path: {args.model_config}")
    print(f"Weights: {args.weights}\n")

    params = TrainingParams(args.config, args.model_config, dataset_folder_override=args.data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = model_factory(params.model_params)
    assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
    print('Loading weights: {}'.format(args.weights))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    stats = evaluate(model, device, params, show_progress=True, log=True)
    print_eval_stats(stats)

    # Esempio per salvare i risultati su file
    # model_name = os.path.splitext(os.path.basename(args.weights))[0]
    # pnv_write_eval_stats("./outputs/pnv_experiment_results.txt", model_name, stats)
    
    print("\nEvaluation complete.")