import numpy as np
import pandas as pd
import scipy.sparse as sp


dtypes = {
    'user': np.int32, 'item': np.int32,
    'rating': np.float32, 'timestamp': np.float64}

class_values = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], dtype=dtypes['rating'])

def get_movies_features(movie_df, item_dict, headers_to_get, num_features):
    item_features = np.zeros((len(item_dict.keys()), num_features), dtype=np.float32)
    for movie_id, g_vec in zip(movie_df['item'].values.tolist(), movie_df[headers_to_get].values.tolist()):
        # check if movie_id was listed in ratings file and therefore in mapping dictionary
        if movie_id in item_dict.keys():
            item_features[item_dict[movie_id], :] = g_vec
    return item_features


def get_user_features(user_idxs):
    user_features = np.zeros((len(set(user_idxs)), 1), dtype=np.float32)
    return user_features


def get_ratings_data(filepath=None, separator=None, dtypes=None):
    return pd.read_csv(
        filepath, sep=separator, header=None,
        names=["user", "item", "rating", "timestamp"], dtype=dtypes, skiprows=1)


def get_movies_data(filepath=None, separator=None, movies_columns_to_drop=None, only_genres=True):
    movie_headers = ['item', 'title', 'genres', 'mean', 'popularity', 'mean_unbiased', 
                    '(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children',
                    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                    'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                    'War', 'Western']

    movie_df = pd.read_csv(filepath, sep=separator, header=None,
                           names=movie_headers, engine='python', encoding='ISO-8859-1', skiprows=1)
    for column_to_drop in movies_columns_to_drop:
        if column_to_drop in movie_df.columns:
            movie_df.drop(column_to_drop, axis=1, inplace=True)
    if only_genres:
        feature_headers = movie_df.columns.values[7:]
    else:
        feature_headers = np.concatenate((movie_df.columns.values[3:5], movie_df.columns.values[7:]))
    num_features = feature_headers.shape[0]
    return movie_df, feature_headers, num_features

def map_data(data):
    """
    From IGMC data_utils.py
    Map data to proper indices in case they are not in a continues [0, N) range

    Parameters
    ----------
    data : np.int32 arrays

    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data

    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)

    return data, id_dict, n


def preprocess_data_to_graph(data_array, testing=False, rating_map=None, post_rating_map=None, ratio=1.0, dtypes=None, class_values=None):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """
    if ratio < 1.0:
        data_array = data_array[data_array[:, -1].argsort()[:int(ratio*len(data_array))]]

    user_nodes_ratings = data_array[:, 0].astype(dtypes['user'])
    item_nodes_ratings = data_array[:, 1].astype(dtypes['item'])
    ratings = data_array[:, 2].astype(dtypes['rating'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    user_nodes_ratings, user_dict, num_users = map_data(user_nodes_ratings)
    item_nodes_ratings, item_dict, num_items = map_data(item_nodes_ratings)

    user_nodes_ratings, item_nodes_ratings, ratings = user_nodes_ratings.astype(np.int64), item_nodes_ratings.astype(np.int32), ratings.astype(np.float64)

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(class_values.tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[user_nodes_ratings, item_nodes_ratings] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(user_nodes_ratings)):
        assert(labels[user_nodes_ratings[i], item_nodes_ratings[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_edges = data_array.shape[0]

    pairs_nonzero = np.array([[u, v] for u, v in zip(user_nodes_ratings, item_nodes_ratings)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    assert(len(idx_nonzero) == num_edges)

    user_idx, item_idx = pairs_nonzero.transpose()

    # create labels
    nonzero_labels = labels[idx_nonzero]

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[idx_nonzero] = labels[idx_nonzero].astype(np.float32) + 1.
    else:
        rating_mx_train[idx_nonzero] = np.array([post_rating_map[r] for r in class_values[labels[idx_nonzero]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    return rating_mx_train, nonzero_labels, user_idx, item_idx, item_dict


def get_item_dict(data_array):
    item_nodes_ratings = data_array[:, 1].astype(dtypes['item'])
    _,item_dict,_ = map_data(item_nodes_ratings)
    return item_dict


def create_dataset_cnn(ratings, top=None):
    if top is not None:
        ratings.groupby('user')['rating'].count()
    
    unique_users = ratings.user.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.user.map(user_to_index)
    
    unique_movies = ratings.item.unique()
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.item.map(movie_to_index)
    
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]
    
    X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y), (user_to_index, movie_to_index)