import numpy as np


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


def complex_accuracy(document, predicted, th=1):
    assert th >= 1
    th = min(len(document), th)
    ground_truth_set = set(document)
    predicted_set = set(predicted)
    return True if len(ground_truth_set & predicted_set) >= th else False


def average_precision(ground_truth, predicted, th=1):
    check_relevant_document = [False] * len(ground_truth)
    check_predicted_document = []

    # Check which prediction is relevant
    for i, single_predicted in enumerate(predicted):
        relevant_predicted = False

        # For each relevant document compute accuracy with the given prediction
        for j, single_relevant in enumerate(ground_truth):
            # if check_relevant_document[j]:
            #     # That document is already relevant
            #     continue

            res = complex_accuracy(single_relevant, single_predicted, th=th)
            if res:
                check_relevant_document[j] = res
                relevant_predicted = True

        check_predicted_document.append(relevant_predicted)
        if np.all(check_relevant_document):
            break

    # Compute precision for each relevant prediction
    precision_array = []
    num_relevant = 0
    for i, res in enumerate(check_predicted_document, 1):
        if res:
            num_relevant += 1
            precision_array.append(num_relevant / i)

    # Add 0 precision for not retrieved document
    for res in check_relevant_document:
        if res is False:
            precision_array.append(0)

    return np.mean(precision_array)


def compute_precision_and_recall(query_results, mode):
    for file in query_results.keys():
        print(f'Computing precision and recall at k for file {file}: ')
        query, ground_truth, predictions = query_results[file]
        print(f'Query: {query}')
        acc = accuracy_k(mode, ground_truth, predictions, k=5)
        print(f"Accuracy at k=5 : {acc}")
        print('\n')


def compute_mean_average_precision(query_results):
    average_precisions = []
    for file in query_results.keys():
        print(f'Computing Average Precision for file {file}: ')
        query, ground_truth, predictions = query_results[file]
        print(f'Query: {query}')
        ap = average_precision(ground_truth, predictions)
        average_precisions.append(ap)
        print(f"Average Precision : {ap}")
        print('\n')
    mean_ap = np.mean(average_precisions)
    print(f"Mean Average Precision : {mean_ap}")
    print('\n')


# if __name__ == '__main__':
    # # Read pickle files with query embedding results
    # results_dir = 'pipeline/query_emb_results/'
    #
    # # Get list of files
    # file_list = [f for f in os.listdir(results_dir) if f.endswith('.pickle')]
    #
    # for file in file_list:
    #     filename = file.replace('.pickle', '')
    #     items = filename.split('_')
    #     print("#" * 80)
    #     if len(items) == 3:
    #         wrapper = items[0]
    #         modality = items[1]
    #         technique = items[2]
    #         print(f"Wrapper: {wrapper}")
    #         print(f"Modality: {modality}")
    #         print(f"Technique: {technique}")
    #     else:
    #         wrapper = items[0]
    #         modality = items[1]
    #         print(f"Wrapper: {wrapper}")
    #         print(f"Modality: {modality}")
    #
    #     with open(results_dir + file, 'rb') as handle:
    #         emb_results = pickle.load(handle)
    #
    #     compute_mean_average_precision(emb_results)
