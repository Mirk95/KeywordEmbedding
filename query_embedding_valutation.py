import os
import numpy as np
import pickle


def precision(ground_truth, predicted, k):
    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted[:k])
    result = len(ground_truth_set & predicted_set) / float(k)
    return result


def recall(ground_truth, predicted, k):
    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted[:k])
    result = len(ground_truth_set & predicted_set) / float(len(ground_truth_set))
    return result


def simple_accuracy(ground_truth, predicted):
    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted)
    return True if len(ground_truth_set & predicted_set) > 0 else False


def accuracy_k(mode, ground_truth, predicted, k):
    if mode == 'singletosingle':
        pred_at_k = simple_accuracy(ground_truth, predicted[:k])
        return pred_at_k
    else:
        pred_at_k = [simple_accuracy(ground_truth, val) for val in predicted[:k]]
        return np.any(pred_at_k)


def compute_precision_and_recall(query_results, mode):
    for file in query_results.keys():
        print(f'Computing precision and recall at k for file {file}: ')
        query, ground_truth, predictions = query_results[file]
        print(f'Query: {query}')
        acc = accuracy_k(mode, ground_truth, predictions, k=5)
        print(f"Accuracy at k=5 : {acc}")
        print('\n')


if __name__ == '__main__':
    # Read pickle files with query embedding results
    results_dir = 'pipeline/query_emb_results/'

    # Get list of files
    file_list = [f for f in os.listdir(results_dir) if f.endswith('.pickle')]

    for file in file_list:
        filename = file.replace('.pickle', '')
        items = filename.split('_')
        print("#" * 80)
        if len(items) == 3:
            wrapper = items[0]
            modality = items[1]
            technique = items[2]
            print(f"Wrapper: {wrapper}")
            print(f"Modality: {modality}")
            print(f"Technique: {technique}")
        else:
            wrapper = items[0]
            modality = items[1]
            print(f"Wrapper: {wrapper}")
            print(f"Modality: {modality}")

        with open(results_dir + file, 'rb') as handle:
            emb_results = pickle.load(handle)

        compute_precision_and_recall(emb_results, modality)
