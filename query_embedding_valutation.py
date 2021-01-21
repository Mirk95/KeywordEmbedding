import os
import pickle
import itertools


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


def compute_precision_and_recall(query_results):
    for file in query_results.keys():
        print(f'Computing precision and recall at k for file {file}: ')
        query, ground_truth, predictions = query_results[file]
        list_of_tuples = []
        list_of_relationships = []
        for gt in ground_truth:
            tuples, relationships = gt[0], gt[1]
            list_of_tuples.append(tuples)
            list_of_relationships.append(relationships)

        list_of_relationships = list(itertools.chain.from_iterable(list_of_relationships))
        if len(list_of_relationships) > 0:
            for rel in list_of_relationships:
                list_of_tuples.append(list(rel))

        actual_values = list(itertools.chain.from_iterable(list_of_tuples))

        print(f'Query: {query}')
        p = precision(actual_values, predictions, k=5)
        print(f'Precision at k=5 : {p}')
        r = recall(actual_values, predictions, k=5)
        print(f'Recall at k=5 : {r}')
        print('\n')


if __name__ == '__main__':
    # Read pickle files with query embedding results
    results_dir = 'pipeline/query_emb_results/'

    # Get list of files
    file_list = [f for f in os.listdir(results_dir) if f.endswith('.pickle')]

    for file in file_list:
        filename = file.replace('.pickle', '')
        items = filename.split('_')
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

        compute_precision_and_recall(emb_results)
