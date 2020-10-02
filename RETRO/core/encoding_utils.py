import numpy as np
import random
from math import *
import scipy.stats, scipy
from word2number import w2n
from bisect import bisect_left

number_embeddings = dict()
embedded_numbers = []
normalization = False
number_dims = 300
buckets = True
standard_deviation = 1.0


def set_normalization(n):
    """
    Sets the normalization flag
    """
    global normalization
    if n is not None:
        normalization = n
    return normalization


def apply_normalization(vec: np.ndarray) -> np.ndarray:
    """
    Normalizes the given vector, if the normalization flag is set
    """
    if normalization:
        return vec / np.linalg.norm(vec)
    else:
        return vec


def set_number_dims(dims):
    """
    Sets the number of dimensions used for encoding numeric values (max. 300)
    """
    global number_dims
    if dims is not None:
        if dims <= 0:
            number_dims = 0
        else:
            number_dims = min(dims, 300)
    return number_dims


def set_buckets(b):
    """
    Sets the buckets flag
    """
    global buckets
    if b is not None:
        buckets = b
    return buckets


def set_standard_deviation(sd):
    """
    Sets the standard deviation used for gaussian encodings
    """
    global standard_deviation
    if sd is not None:
        standard_deviation = sd
    return standard_deviation


def text_to_vec(term, vec_bytes, terms, tokenization_settings):
    """
    Encodes a term to a 300-dimensional vector using the word embeddings given in "terms"

    :return: a boolean that indicates, if the vector was inferred and
        the vector itself in the form vector.tobytes()
    """
    tokenization_strategy = tokenization_settings['TEXT_TOKENIZATION']
    if vec_bytes is not None:
        return False, vec_bytes
    else:
        if term is None:
            return True, None

        splits = [x.replace('_', '') for x in term.split('_')]
        i = 1
        j = 0
        current = [terms, None, -1]
        vector = None
        last_match = (0, None, -1)
        count = 0
        while i <= len(splits) or last_match[1] is not None:
            subword = '_'.join(splits[j:i])
            if subword in current[0]:
                current = current[0][subword]
                if current[1] is not None:
                    last_match = (i, np.fromstring(bytes(current[1]), dtype='float32'), current[2])
            else:
                if last_match[1] is not None:
                    if vector is not None:
                        if tokenization_strategy == 'log10':
                            vector += last_match[1] * np.log10(last_match[2])
                            count += np.log10(last_match[2])
                        else:  # 'simple' or different
                            vector += last_match[1]
                            count += 1
                    else:
                        if tokenization_strategy == 'log10':
                            vector = last_match[1] * \
                                     np.log10(last_match[2])
                            count += np.log10(last_match[2])
                        else:  # 'simple' or different
                            vector = last_match[1]
                            count += 1
                    j = last_match[0]
                    i = j
                    last_match = (0, None, -1)
                else:
                    j += 1
                    i = j
                current = [terms, None, -1]
            i += 1
        if vector is not None:
            vector /= count
            return True, vector.tobytes()
        else:
            return True, None


def num_to_vec_one_hot(num, min_value, max_value, column_vec):
    """
    Encodes a number to a 300-dimensional vector using a one-hot encoding

    -> divides the range [min_value, max_value] in number_dims equally spaced sub-ranges, that are used for encoding

    If column_vec is not None, the centroid of the number and the column name will be calculated.

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / number_dims
        index = int((num - min_value) // range_size)
        return bucket_to_vec_one_hot(min(number_dims-1, index), column_vec)
    else:
        return np.zeros(300, dtype='float32')


def bucket_to_vec_one_hot(bucket, column_vec):
    """
    Encodes a bucket index to a 300-dimensional vector using one-hot encoding with the values -1.0 and 1.0.

    If column_vec is not None, the centroid of the number and the column name will be calculated.

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        vec[0:number_dims] = -1.0
        vec[bucket] = 1.0
        if column_vec is not None:
            cv = np.frombuffer(column_vec, dtype='float32')
            vec += cv
            vec /= 2
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def num_to_vec_one_hot_gaussian(num, min_value, max_value):
    """
    Encodes a number to a 300-dimensional vector using a one-hot encoding with a gaussian filter

    -> divides the range [min_value, max_value] in number_dims equally spaced sub-ranges, that are used for encoding

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / number_dims
        index = int((num - min_value) // range_size)
        return bucket_to_vec_one_hot_gaussian(min(number_dims-1, index))
    else:
        return np.zeros(300, dtype='float32')


def num_to_vec_one_hot_gaussian_fluent(num, min_value, max_value):
    """
    Encodes a number to a 300-dimensional vector using a one-hot encoding with a gaussian filter

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        value_range = max_value - min_value
        norm_num = ((num - min_value) / value_range) * number_dims
        vec = np.zeros(300, dtype='float32')
        for x in range(number_dims):
            vec[x] = gaussian(x * 0.5, norm_num * 0.5)
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32')


def bucket_to_vec_one_hot_gaussian(bucket):
    """
    Encodes a bucket index to a 300-dimensional vector using one-hot encoding with a gaussian filter

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        for x in range(number_dims):
            vec[x] = gaussian(x * 0.5, bucket * 0.5)  # the factor 0.5 streches the function
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def num_to_vec_we_regression(num):
    """
    Encodes a number to a 300-dimensional vector using word embeddings of nearby numbers.

    :return: vector in the form vector.tobytes()
    """
    if number_embeddings.get(num) is not None:
        return number_embeddings[num]
    else:
        closest_numbers = get_neighbors(embedded_numbers, num)
        result = np.zeros(300, dtype='float32')
        for i in closest_numbers:
            vec = np.frombuffer(number_embeddings[i], dtype='float32')
            result += vec
        result /= len(closest_numbers)
        return apply_normalization(result).tobytes()


def num_to_vec_unary(num, min_value, max_value, column_vec):
    """
    Encodes a number to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    -> divides the range [min_value, max_value] in number_dims equally spaced sub-ranges, that are used for encoding

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / number_dims
        index = int((num - min_value) // range_size)
        return bucket_to_vec_unary(min(number_dims-1, index), column_vec)
    else:
        return np.zeros(300, dtype='float32').tobytes()


def bucket_to_vec_unary(bucket, column_vec):
    """
    Encodes a bucket index to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        if bucket < number_dims-1:
            vec[bucket + 1:number_dims] -= 1.0
        vec[0:bucket + 1] = 1.0
        if column_vec is not None:
            cv = np.frombuffer(column_vec, dtype='float32')
            vec += cv
            vec /= 2
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def num_to_vec_unary_gaussian(num, min_value, max_value):
    """
    Encodes a number to a 300-dimensional vector using a unary encoding with a gaussian filter

    -> divides the range [min_value, max_value] in number_dims equally spaced sub-ranges, that are used for encoding

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / number_dims
        index = int((num - min_value) // range_size)
        return bucket_to_vec_unary_gaussian(min(number_dims-1, index))
    else:
        return np.zeros(300, dtype='float32')


def num_to_vec_unary_gaussian_fluent(num, min_value, max_value):
    """
    Encodes a number to a 300-dimensional vector using a unary encoding with a gaussian filter

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        value_range = max_value - min_value
        norm_num = ((num - min_value) / value_range) * number_dims
        norm_int = floor(norm_num)
        vec = np.zeros(300, dtype='float32')
        vec[0: norm_int+1] = 1.0
        for x in range(norm_int+1, number_dims):
            vec[x] = gaussian(x * 0.5, norm_num * 0.5)
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32')


def bucket_to_vec_unary_gaussian(bucket):
    """
    Encodes a bucket index to a 300-dimensional vector using unary encoding with a gaussian filter

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        vec[0: bucket+1] = 1.0
        for x in range(bucket+1, number_dims):
            vec[x] = gaussian(x * 0.5, bucket * 0.5)  # the factor 0.5 streches the function
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def num_to_vec_unary_column_partial(num, min_value, max_value, column_vec):
    """
    Encodes a number to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    -> divides the range [min_value, max_value] in number_dims equally spaced sub-ranges, that are used for encoding

    -> the 300 - number_dims last vector values are used to represent the column name word embedding

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / number_dims
        index = int((num - min_value) // range_size)
        return bucket_to_vec_unary_column_partial(min(number_dims - 1, index), column_vec)
    else:
        return np.zeros(300, dtype='float32').tobytes()


def bucket_to_vec_unary_column_partial(bucket, column_vec):
    """
    Encodes a bucket index to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    -> the 300 - number_dims last vector values are used to represent the column name word embedding

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        vec[0:bucket + 1] = 1.0
        vec[bucket + 1:number_dims] = -1.0
        cv = np.frombuffer(column_vec, dtype='float32')
        for i in range(number_dims, 300):
            vec[i] = column_vec[i]
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def num_to_vec_unary_random_dim(num, min_value, max_value, column_vec):
    """
    Encodes a number to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    -> divides the range [min_value, max_value] in number_dims equally spaced sub-ranges, that are used for encoding

    -> distributes the dimensions used for encoding randomly (using the column_vec as a seed)

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / number_dims
        index = int((num - min_value) // range_size)
        return bucket_to_vec_unary_random_dim(min(number_dims - 1, index), column_vec)
    else:
        return np.zeros(300, dtype='float32').tobytes()


def bucket_to_vec_unary_random_dim(bucket, column_vec):
    """
    Encodes a bucket index to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    -> randomly distributes the dimensions used for encoding (using the column_vec as a seed)

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        vec[0:bucket + 1] = 1.0
        vec[bucket + 1:number_dims] = -1.0
        random.seed(column_vec)
        random.shuffle(vec.flat)
        return apply_normalization(vec).tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def generate_random_vec():
    """
    Generates a random 300-dimensional vector, that is randomly filled with values between -1.0 and 1.0

    :return: vector in the form vector.tobytes()
    """
    vec = np.zeros(300, dtype='float32')
    for i in range(number_dims):
        vec[i] = (np.random.random() * 2) - 1.0
    return apply_normalization(vec).tobytes()


def initialize_numeric_word_embeddings(cur, we_table_name):
    """
    Traverses all word embeddings that represent numbers and saves them to number_embeddings.

    This function should be called when using the we-regression numeric tokenization strategy
    """
    if len(number_embeddings.keys()) == 0:
        print('Initializing numeric word embeddings')
        w2n.print_function = None
        query = 'SELECT word::varchar, vector FROM %s' % we_table_name
        cur.execute(query)
        result = dict()
        count = 0
        for word, vector in cur.fetchall():
            try:
                number = w2n.word_to_num(word)
                if number is not None:
                    if result.get(number) is None:
                        result[number] = []
                    result[number].append(vector)
                    count += 1
            except ValueError:
                pass
        print("Numeral word embeddings found: %s" % count)
        for key in result.keys():
            vectors = list(map(lambda x: np.frombuffer(x, dtype='float32'), result[key]))
            final_vec = np.zeros(300, dtype='float32')
            for vec in vectors:
                final_vec += vec
            final_vec /= len(vectors)
            number_embeddings[key] = final_vec.tobytes()
            embedded_numbers.append(key)
        embedded_numbers.sort()


def get_neighbors(sorted_list, value):
    """
    Returns the neighboring value(s) to value, that are contained in sorted_list.

    If two numbers are equally close, return the smallest number.

    Source: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    """
    pos = bisect_left(sorted_list, value)
    if pos == 0:
        return [sorted_list[0]]
    if pos == len(sorted_list):
        return [sorted_list[-1]]
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    return [before, after]


def bucket_valid(bucket):
    """
    checks if given bucket index is inside the valid range [0, number_dims)
    """
    return 0 <= bucket < number_dims


def needs_column_encoding(mode):
    """
    Returns True, if an encoding mode needs a column word embedding vector, otherwise False
    """
    return mode in ["one-hot-column-centroid",
                    "unary-column-centroid",
                    "unary-column-partial",
                    "unary-random-dim"]


def needs_min_max_values(mode, buckets):
    """
    Returns True, if an encoding mode needs minimum and maximum column values, otherwise False
    """
    return not buckets and mode in ['one-hot',
                                    'one-hot-gaussian',
                                    'one-hot-gaussian-fluent',
                                    'unary',
                                    'unary-gaussian',
                                    'unary-gaussian-fluent']


def gaussian(x, mean):
    return sqrt(2*pi) * standard_deviation * scipy.stats.norm.pdf(x, mean, standard_deviation)
