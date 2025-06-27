# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import random
import tqdm
import sys
import copy

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader

PERCENTAGE = 0.1
num_neighbors = 10


def evaluate(model,
             device,
             params: TrainingParams,
             log: bool = False,
             show_progress: bool = False):
    eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
                           'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']

    eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
                        'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']

    # eval_database_files = ['business_evaluation_database.pickle']
    # eval_query_files = ['business_evaluation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        # print('Query sets: {}'.format(query_sets))
        # print('Database sets: {}'.format(database_sets))

        temp = evaluate_dataset(model,
                                device,
                                params,
                                database_sets,
                                query_sets,
                                log=log,
                                show_progress=show_progress)
        for it, elem in enumerate(temp):
            perc = str(100 - (it * (PERCENTAGE * 100))) + "%"
            stats[database_file + perc] = elem

    return stats


def removing_sample(k,
                    n,
                    mean_set,
                    var_set,
                    query_set):
    print("Removing: " + str(k * (PERCENTAGE * 100)) + "%")
    mean_set_ = copy.deepcopy(mean_set)
    query_set_ = copy.deepcopy(query_set)
    index_dict_array = []

    for m in range(n):
        sum_var = np.sum(var_set[m], axis=1)
        index_inv = np.argsort(sum_var)[::-1]
        top_index = int(k * PERCENTAGE * len(var_set[m]))
        index_t = index_inv[:top_index]

        # take the indexes that are not in the top 25%
        index_not_t = np.setdiff1d(np.arange(len(mean_set[m])), index_t)

        # remove the indexes from the embeddings
        mean_set_[m] = np.array(mean_set[m])[index_not_t]

        query_set_[m] = {i: query_set[m][key] for i, key in enumerate(index_not_t)}

        index_dict = {}
        for i in range(len(mean_set_[m])):
            index_dict[i] = index_not_t[i]
        index_dict_array.append(index_dict)

    return mean_set_, query_set_, index_dict_array


def evaluate_dataset(model,
                     device,
                     params: TrainingParams,
                     database_sets,
                     query_sets,
                     log: bool = False,
                     show_progress: bool = False):
    database_embeddings_mean = []
    query_embeddings_mean = []
    database_embeddings_var = []
    query_embeddings_var = []

    model.eval()
    # n_models = len(models)

    # DATABASE
    cont = 0
    for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        # if cont == 2:
        #     break
        # cont+=1
        emb_per_model = []
        # for each model, compute the embedding
        # than aggregate the embeddings with mean and variance
        # for model in models:
        #     emb_per_model.append(get_latent_vectors(model, set, device, params))

        emb, epistemic = get_latent_vectors(model, set, device, params)

        emb_mean = emb
        emb_var = epistemic

        # for i in range(len(emb_per_model[0])):
        #     emb_mean.append(np.mean([emb_per_model[j][i] for j in range(n_models)], axis=0))
        #     emb_var.append(np.var([emb_per_model[j][i] for j in range(n_models)], axis=0))
        database_embeddings_mean.append(emb_mean)
        database_embeddings_var.append(emb_var)

    # QUERY
    cont = 0
    for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        # if cont == 2:
        #     break
        # cont+=1
        emb_per_model = []
        # for model in models:
        #     emb_per_model.append(get_latent_vectors(model, set, device, params))
        emb, epistemic = get_latent_vectors(model, set, device, params)

        emb_mean = emb
        emb_var = epistemic
        # emb_mean = []
        # emb_var = []
        # for i in range(len(emb_per_model[0])):
        #     emb_mean.append(np.mean([emb_per_model[j][i] for j in range(n_models)], axis=0))
        #     emb_var.append(np.var([emb_per_model[j][i] for j in range(n_models)], axis=0))
        query_embeddings_mean.append(emb_mean)
        query_embeddings_var.append(emb_var)

    # with open('mean_and_var/database_embeddings_mean.pickle', 'rb') as f:
    #     database_embeddings_mean = pickle.load(f)
    #
    # with open('mean_and_var/database_embeddings_var.pickle', 'rb') as f:
    #     database_embeddings_var = pickle.load(f)
    #
    # with open('mean_and_var/query_embeddings_mean.pickle', 'rb') as f:
    #     query_embeddings_mean = pickle.load(f)
    #
    # with open('mean_and_var/query_embeddings_var.pickle', 'rb') as f:
    #     query_embeddings_var = pickle.load(f)
    del model

    stats = []
    recall = np.zeros(num_neighbors)
    count = 0
    one_percent_recall = []
    for k in range(0, int(1 / PERCENTAGE)):
        index_dict_array = []
        len_database_set = len(database_sets)
        len_query_set = len(query_sets)

        # database_embeddings_mean_, index_dict_array = removing_sample(k, len_database_set, database_embeddings_mean, database_embeddings_var)
        # query_embeddings_mean_, _ = removing_sample(k, len_query_set, query_embeddings_mean, query_embeddings_var)
        query_embeddings_mean_, query_sets_, _ = removing_sample(k, len_query_set, query_embeddings_mean,
                                                                 query_embeddings_var, query_sets)
        for i in range(len(query_sets)):
            for j in range(len(query_sets)):
                if i == j:
                    continue
                # pair_recall, pair_opr = get_recall(i, j, index_dict_array, database_embeddings_mean_, query_embeddings_mean, query_embeddings_var, query_sets,
                #                                 database_sets, log=log)
                pair_recall, pair_opr = get_recall(i, j, index_dict_array, database_embeddings_mean,
                                                   query_embeddings_mean_, query_embeddings_var, query_sets_,
                                                   database_sets, log=log)
                recall += np.array(pair_recall)
                count += 1
                one_percent_recall.append(pair_opr)

        ave_recall = recall / count
        ave_one_percent_recall = np.mean(one_percent_recall)
        stats_temp = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
        stats.append(stats_temp)
    return stats


def get_latent_vectors(model,
                       set,
                       device,
                       params: TrainingParams):
    # Adapted from original PointNetVLAD code

    if params.debug:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    pc_loader = PNVPointCloudLoader()

    model.eval()
    embeddings = None
    epistemics = None
    for i, elem_ndx in enumerate(set):
        pc_file_path = os.path.join(params.dataset_folder, set[elem_ndx]["query"])
        pc = pc_loader(pc_file_path)
        pc = torch.tensor(pc)

        prediction = compute_embedding(model, pc, device, params)
        embedding = prediction['global'].detach().cpu().numpy()
        gamma = prediction['gamma'].detach().cpu().numpy()
        embedding = gamma
        nu = prediction['nu'].detach().cpu().numpy()
        alpha = prediction['alpha'].detach().cpu().numpy()
        beta = prediction['beta'].detach().cpu().numpy()

        epistemic = beta / (nu * (alpha - 1))

        if embeddings is None:
            embeddings = np.zeros((len(set), embedding.shape[1]), dtype=embedding.dtype)
            epistemics = np.zeros((len(set), epistemic.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding
        epistemics[i] = epistemic

    return embeddings, epistemics


def compute_embedding(model, pc, device, params: TrainingParams):
    """
    COMPUTE THE EMBEDDINGS FOR A SINGLE POINT CLOUD
    """
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        # Compute global descriptor
        prediction = model(batch)

    return prediction


def get_recall(m,
               n,
               index_dict_array,
               database_vectors,
               query_vectors,
               query_var,
               query_sets,
               database_sets,
               log=False):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    queries_var = query_var[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    recall = [0] * num_neighbors

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    # index_dict_array_list = list(index_dict_array[n].values())
    for i in range(len(queries_output)):
        # i is query element ndx
        # filtered_index = index_dict_array_list[i]
        query_details = query_sets[n][i]  # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue

        # map the indices to the original indices
        # true_neighbors_mapped = []
        # for indx in true_neighbors:
        #     if indx in index_dict_array[m]:
        #         true_neighbors_mapped.append(index_dict_array[m][indx])
        # if len(true_neighbors_mapped) == 0:
        #     continue

        num_evaluated += 1

        # VI = np.linalg.inv(np.diag(queries_var[i]))
        # nn = NearestNeighbors(n_neighbors=num_neighbors,
        #                       algorithm='auto',
        #                       metric='mahalanobis',
        #                       metric_params={'VI': VI}, n_jobs=-1).fit(database_output)
        #                       )

        # Find nearest neighbours
        indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors, return_distance=False)
        # print(indices)
        # indices = nn.Kneighbors(np.array([queries_output[i]]), return_distance=False)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100

    return recall, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall']))
        print(stats[database_name]['ave_recall'])


def pnv_write_eval_stats(file_name, stats):
    with open(file_name, 'a') as f:
        # write only the average recall with the name of the database
        for database_name in stats:
            f.write('Dataset: {}\n'.format(database_name))
            f.write(str(stats[database_name]['ave_recall']))
            f.write('\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true')
    parser.set_defaults(log=False)

    args = parser.parse_args()
    # make an array of weight from the given path
    weights = args.weights

    # for weight in os.listdir(args.weights):
    #     weights.append(weight)

    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Weights: {}'.format(weights))
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    print('')

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params.model_params)
    # fill the array "m" with the files in the given path
    # if weights is not None:
    #     assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
    #     print('Loading weight: {}'.format(weights[0]))
    #     model.load_state_dict(torch.load(os.path.join(args.weights, weights[0]), map_location=device))

    assert os.path.isfile(weights), 'Cannot open network weights: {}'.format(weights)
    print('Loading weight: {}'.format(weights))
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)

    stats = evaluate(model, device, params, args.log, show_progress=True)
    print_eval_stats(stats)

    # Save results to the text file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(args.weights)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    # pnv_write_eval_stats("results/pnv_experiment_results_normal.txt",
    #                      "results/pnv_experiment_recalls_normal.txt",
    #                      prefix, stats)
    pnv_write_eval_stats("eval/results/pnv_experiment_results_DER_based_triplet_loss.txt", stats)
