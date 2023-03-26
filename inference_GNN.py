from torch_geometric.loader import DataLoader
from recommender_utils import *
import torch
from model_GNN import IGMC
import pandas as pd
import numpy as np
import sys
import random
from IGMC.data_utils import *
from IGMC.util_functions import *
from IGMC.preprocessing import *


def get_movie_ids_to_predict(user_ratings, all_movies):
    all_movie_ids = all_movies['item'].unique()
    user_rating_ids = user_ratings['item'].unique()
    return [idx for idx in all_movie_ids if idx not in user_rating_ids]


def populate_with_dummy_graphs(inference_data, movies_to_predict):
    timestamp = 945173447
    dummy_review = 2.0
    user_min_idx = all_data['user'].max() + 1
    graphs_for_inference = inference_data.iloc[:0,:].copy()
    for i, movie_id in enumerate(movies_to_predict):
        dummy_inference_graph = inference_data.copy()
        user_idx = user_min_idx + i
        dummy_inference_graph['user'] = user_idx
        list_row = [user_idx, movie_id, dummy_review, timestamp]
        dummy_inference_graph.loc[len(dummy_inference_graph)] = list_row
        graphs_for_inference = pd.concat([graphs_for_inference, dummy_inference_graph], ignore_index=True)
    return graphs_for_inference


def add_additional_row_for_dataloader(df, i):
    list_row = [i, 0, 3.0, 999999999]
    df.loc[len(df)] = list_row


def get_movie_name_by_id(df, id):
    return df.loc[df['item'] == id]['title'].item()


def get_movie_popularity_by_id(df, id):
    return df.loc[df['item'] == id]['popularity'].item()


def print_best_N_predictions(predictions, movies, n=10):
    i = 0
    d_view = [(v, k) for k, v in predictions.items()]
    d_view.sort(reverse=True)
    print(f"Best {n} predictions by possible review:")
    for v, k in d_view:
        if i == n:
            break
        print(f"idx {k}; {get_movie_name_by_id(movies, k)}: {v}")
        i += 1


def print_best_N_predictions_on_popularity(predictions, movies, n=10):
    i = 0
    d_view = [(v, k) for k, v in predictions.items()]
    d_view.sort(reverse=True)
    print(f"Best {n} predictions by possible review and popularity:")
    for v, k in d_view:
        if i == n:
            break
        popularity = get_movie_popularity_by_id(movies, k)
        if random.random() < popularity:
            print(f"idx {k}; {get_movie_name_by_id(movies, k)}: {v}")
            i += 1


def get_trained_model(model_path, n_side_features=0):
    if not n_side_features:
        model_load = IGMC()
    else:
        model_load = IGMC(side_features=True, n_side_features=n_side_features)
    model_load.load_state_dict(torch.load(model_path))
    model_load.eval()
    return model_load


def get_all_ratings_data():
    data_train = get_ratings_data(filepath='data_small/train.csv', separator=r',', dtypes=dtypes)
    data_val = get_ratings_data(filepath='data_small/validate.csv', separator=r',', dtypes=dtypes)
    data_test = get_ratings_data(filepath='data_small/test.csv', separator=r',', dtypes=dtypes)
    return pd.concat([data_train, data_test, data_val], ignore_index=True)


def get_predictions(model, dataloader, movies_to_predict):
    predictions = dict()
    batches_for_prediction = len(movies_to_predict)
    device = torch.device("cpu")
    for i, test_batch in enumerate(dataloader):
        print(f"{i}/{batches_for_prediction - 1}")
        test_batch = test_batch.to(device)
        with torch.no_grad():
            y_my_pred = model(test_batch)
        torch.cuda.empty_cache()
        predictions[movies_to_predict[i]] = y_my_pred[-1].item()
        if i == batches_for_prediction - 1:
            break
    return predictions


if __name__ == '__main__':
    # input_data_path = 'my_reviews/1.csv'
    # model_path = 'models/graph_80epochs_wo_features.pt'
    use_features = False

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        input_data_path = sys.argv[2]
    
    if len(sys.argv) > 3:
        use_features = True if sys.argv[3] == "1" else False
    
    if len(sys.argv) > 4:
        only_genres = True if sys.argv[4] == "1" else False

    start_time = time.time()
    all_data = get_all_ratings_data()
    if only_genres:
        all_movies, genre_headers, num_genres = get_movies_data(filepath='data_small/movies.csv', separator=r',', movies_columns_to_drop=['genres'])
    else:
        all_movies, genre_headers, num_genres = get_movies_data(filepath='data_small/movies.csv', separator=r',', movies_columns_to_drop=['genres'], only_genres=False)

    inference_data = get_ratings_data(filepath=input_data_path, separator=r',', dtypes=dtypes)
    input_length = len(inference_data)
    movies_to_predict = get_movie_ids_to_predict(inference_data, all_movies)

    print("Creating dummy graphs...")
    dummy_graphs_for_inference = populate_with_dummy_graphs(inference_data, movies_to_predict)
    print("Graphs created. Concating dummy graphs with main data graph...")
    graph_for_inference = pd.concat([dummy_graphs_for_inference, all_data], ignore_index=True)
    print("Done")

    # add_additional_row_for_dataloader(graphs_for_inference, item_indices_to_predict[-1] + 1)
    print("Preprocessing data for dataset...")
    my_data_array = np.array(graph_for_inference.values.tolist())
    my_adjacency_mx, my_labels, my_user_idx, my_item_idx, my_item_dict = preprocess_data_to_graph(my_data_array, dtypes=dtypes, class_values=class_values)
    print("Done")

    item_features_array = None
    user_features_array = None
    if use_features:
        print("Creating features matrix...")
        item_features = sp.csr_matrix(get_movies_features(all_movies, my_item_dict, genre_headers, num_genres))
        user_features = sp.csr_matrix(get_user_features(my_user_idx))
        item_features_array = item_features.toarray()
        user_features_array = user_features.toarray()
        print("Done")

    print("Creating dataset...")
    my_dataset = MyDynamicDataset(root='data_test/processed/test', A=my_adjacency_mx, 
    links=(my_user_idx, my_item_idx), labels=my_labels, h=1, sample_ratio=1.0, 
    max_nodes_per_hop=200, u_features=user_features_array, v_features=item_features_array, class_values=class_values)
    print("Done")
    
    print("Creating dataloader...")
    my_data_loader = DataLoader(my_dataset, input_length + 1, shuffle=False, num_workers=1)
    print(f"Done. Batches in dataloader: {len(my_data_loader)}. Batches actually used in inference: {len(movies_to_predict)}")
    
    print("Reading model from disk...")
    model = get_trained_model(model_path=model_path, n_side_features=item_features.shape[1])
    print("Done")
    
    print("Starting inference")
    predictions = get_predictions(model, my_data_loader, movies_to_predict)
    print("Inference done")
    
    print(get_movie_name_by_id(all_movies, max(predictions, key=predictions.get)))
    print_best_N_predictions(predictions, all_movies)
    print_best_N_predictions_on_popularity(predictions, all_movies)
    print(f"Took {int((time.time() - start_time) / 60)}m{int((time.time() - start_time) % 60)}s")