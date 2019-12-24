"""Classification System for letters originating from
scanning the pages of a book.

The aim of this project was to develop an OCR
system from the ground up mainly using Numpy.
By prohibiting and constraining the classifier
from using more than 10 dimensions, the main challenge
was developing a strong dimensionality reduction
component. 

Author: Raivo Koot
Date: 17 December 2019

version: v1.0
"""
import difflib
import numpy as np
import utils.utils as utils
import scipy.linalg
from scipy.stats import mode
import imageio
from PIL import Image
from PIL import ImageFilter
from PIL.ImageFilter import (
    GaussianBlur
    )

"""
Constants and hyperparameters

ORIGINAL_SAMPLE_DIMENSIONS_2D - The shape of the original letter vectors if reshaped into the rectangular image
                                in order to, for example, view them in a plot

NUM_PRINCIPAL_COMPONTENTS - The amount of principal component axes to compute

CROP_WHITE_BB - A uniform bounding box for all letters for
                cropping excessive whitespace from the sides
                of the letter images

NEW_DIMENSIONS - The new number of dimensions leftover after
                 cropping whitespace from the images with the
                 above mentioned bounding box

DARKNESS_THRESHOLD - Used for simplifying the data by making all pixel
                     values above this threshold a 0 and all below a 1

NEAREST_NEIGHBOURS - The number of neighbors for the k-nearest neighbour classifier

SPECK_REMOVAL_ITERATIONS - The amount of times to perform speck removal.
                           Applied on very noisy letters.

MAXIMUM_NUMBER_WHITE_NEIGHBORS - The maximum number of white pixel neighbours
                                 to still stay a black pixel when performing
                                 speck removal
MAXIMUM_NEIGHBOUR_THRESHOLD - A threshold that is derived from MAXIMUM_NUMBER_WHITE_NEIGHBORS
                            - because the convolution performing speck removal
                            - divides the neighbour count by 9.

"""

ORIGINAL_SAMPLE_DIMENSIONS_2D = (39, 60)
CROP_WHITE_BB = (0,0,35,30)
NEW_DIMENSIONS_2D = (30,35)
NEW_DIMENSIONS = 1050
IMAGE_CENTER = np.array([14,17])

DARKNESS_THRESHOLD = 0.5
NUM_PRINCIPAL_COMPONTENTS = 10
NEAREST_NEIGHBOURS = 7

SPECK_REMOVAL_ITERATIONS = 2
MAXIMUM_NUMBER_WHITE_NEIGHBORS = 5
MAXIMUM_NEIGHBOUR_THRESHOLD = 0.6

WORD_FILE_NAMES = ['top_hundred_words.txt', 'top_thousand_words.txt', \
                   'top_tenthousand_words.txt', 'top_sixtythousand_words.txt']

CROP_WHITESPACE = True
MIN_MAX_SCALE = True
MAKE_VALUES_BINARY = True
APPLY_BLURRING = False
APPLY_SPECK_REMOVAL = True
APPLY_IMAGE_CENTERING = False
CORRECT_LABELS = True
SHOW_DEBUG_LETTERS = False


def reduce_dimensions(feature_vectors_full, model):
    """Performs dimensionality reduction and data-cleaning.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    if CROP_WHITESPACE:
        feature_vectors_full = crop_whitespace(feature_vectors_full)

    if APPLY_IMAGE_CENTERING:
        feature_vectors_full = center_images(feature_vectors_full)

    if MIN_MAX_SCALE:
        feature_vectors_full = applyMinMaxScaling(feature_vectors_full)
    if MIN_MAX_SCALE and MAKE_VALUES_BINARY:
        make_values_binary(feature_vectors_full)

    if APPLY_BLURRING or APPLY_SPECK_REMOVAL:
        feature_vectors_full = remove_specks_or_blur(feature_vectors_full)

    if SHOW_DEBUG_LETTERS:
        save_X_to_images(feature_vectors_full, NEW_DIMENSIONS_2D)



    if 'pcaTransformer' in model.keys():
        pcaTransformer = model['pcaTransformer']
    else:
        pcaTransformer = findPCATransformer(feature_vectors_full)
        model['pcaTransformer'] = pcaTransformer.tolist()

    X = reduceToPrincipalComponents(feature_vectors_full, pcaTransformer)

    return X

def applyMinMaxScaling(X):
    """ Applies min-max scaling to each row in the given feature vector matrix.
    Uses a slightly adjusted version of min-max scaling
    where a pixel is rescaled in respect to values of the
    other pixels of the image, instead of values of the same pixel
    from all other images

    Params:
    X - feature vectors of image pixel values stored as rows in a matrix
    """
    mins = np.amin(X, axis=1)
    maxs = np.amax(X, axis=1)

    # Take care of corrupted images to avoid zero divisions in other
    # functions as a result of scaling
    faulty_image_indices = np.argwhere(np.subtract(maxs, mins) == 0)


    # Turn min and max lists into column vectors
    mins = np.reshape(mins, (mins.shape[0],1))
    maxs = np.reshape(maxs, (maxs.shape[0],1))


    max_minus_min = np.subtract(maxs, mins)
    # Correct zero values of broken images so that below
    # scaling does not cause a division by zero
    max_minus_min[faulty_image_indices, 0] = 1


    X_scaled = (np.subtract(X, mins) / max_minus_min)

    return X_scaled

def make_values_binary(X, threshold=DARKNESS_THRESHOLD):
    """ Reduces complexity by simply giving pixels
    below a specific darkness threshold a value of 0
    and pixels above a 1. Assumes the pixels have been
    scaled to have a value betwen zero and one.

    Params:
    X - feature vectors of image pixel values stored as rows in a matrix
    """
    light_pixel_indices = np.argwhere(X.flatten() > threshold)
    dark_pixel_indices = np.argwhere(X.flatten() <= threshold)

    np.put(X, light_pixel_indices, 1)
    np.put(X, dark_pixel_indices, 0)

def crop_whitespace(X):
    """
    Returns the data matrix X where every sample is cut down to have less
    whitespace to the right and bottom.

    Params:
    X - feature vector matrix
    """
    cropped_X = np.empty((X.shape[0], NEW_DIMENSIONS))

    for i in range(len(X)):
        rectangular_image_array = np.reshape(X[i,:], ORIGINAL_SAMPLE_DIMENSIONS_2D)
        image = Image.fromarray(rectangular_image_array)

        image = image.crop(CROP_WHITE_BB)

        image_array = np.array(image)
        image_array = np.reshape(image_array, (NEW_DIMENSIONS,))
        cropped_X[i,:] = image_array

    return cropped_X

def reduceToPrincipalComponents(data, transformer):
    """
    Project data onto principal component axes.

    Params:
    data - feature vector matrix
    transformer - the linear transformation matrix used for pca
    """
    v = transformer

    # compute the mean vector
    datamean = np.mean(data)

    # subtract mean from all data points
    centered = data - datamean

    # project points onto PCA axes
    x = np.dot(centered, v)

    return x[:,:]


def findPCATransformer(data, n=NUM_PRINCIPAL_COMPONTENTS):
    """
    Given data, find the principal component axes.

    Params:
    data - feature vector matrix
    n - the amount of principal component axes to find
    """

    # compute data covariance matrix
    covx = np.cov(data, rowvar=0)
    # compute first N pca axes
    n_orig = covx.shape[0]
    [d, v] = scipy.linalg.eigh(covx, eigvals=(n_orig - n, n_orig - 1))
    v = np.fliplr(v)


    return v

def remove_specks_or_blur(X):
    """This attempts to reduce the noise introduced in
       the later pages that are harder to read.
       It does this by attempting to make any black pixel
       white that does not have enough black neighbour pixels.

       Alternatively, it applies blurring to the image.

    Params:
    X - feature vectors stored as rows in a matrix
    """
    speck_removal_convolution = (
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
    )

    convolution = ImageFilter.Kernel(
    size=(3,3),
    kernel=speck_removal_convolution,
    scale=9,
    offset=0
    )

    # Apply to each letter individually
    for i in range(len(X)):

        # Reshape the feature vector to image dimensions and convert the object
        # to a Pil Image
        rectangular_image_array = np.reshape(X[i,:], NEW_DIMENSIONS_2D)
        image = Image.fromarray(rectangular_image_array)
        image = image.convert("L")

        if APPLY_BLURRING:
            image = image.filter(GaussianBlur(radius=2))
            #image = image.filter(ImageFilter.BLUR)

        elif APPLY_SPECK_REMOVAL:
            image = image.filter(convolution)

        # Convert the Pil Image back to a numpy feature vector
        image_array = np.array(image)
        image_array = np.reshape(image_array, (NEW_DIMENSIONS,))
        X[i,:] = image_array

    return X

def find_distances_from_center(X):
    """ Given a 2D numpy array of a letter, finds the
        average row and column index of the black pixels (Or
        in other words, the center of mass of the letter). It then returns
        the row and column distance that this letter is away
        from the center of the image.

    Params:
    letter - a 2D matrix representing a letter
    """

    # 3D coordinates: letter, row, column
    black_pixel_coordinates = np.argwhere(X <= DARKNESS_THRESHOLD)

    # A row for each letter and a column x and y offset
    mass_centers = np.zeros(shape=(X.shape[0], 2), dtype=int)

    # Faster but less understandable
    # averages = [np.average(black_pixel_coordinates[black_pixel_coordinates[:,0] == letter_index][:,[1,2]], axis=0).round().astype(int) for letter_index in range(X.shape[0])]

    # For each letter find average coordinates
    for i in range(X.shape[0]):
        letter_black_coordinates = black_pixel_coordinates[black_pixel_coordinates[:,0] == i][:,[1,2]]
        letter_center = np.average(letter_black_coordinates, axis=0).round().astype(int)

        mass_centers[i,:] = letter_center

    return IMAGE_CENTER - mass_centers

def center_images(X):
    """ For each image in the given matrix, shifts the entire image so
        that its center of mass (black pixels) lies at the center of the image.

    Params:
    X - a matrix containing feature vectors
    """

    X_2d = np.reshape(X, (X.shape[0],) + NEW_DIMENSIONS_2D)

    offsets = find_distances_from_center(X_2d)

    for i in range(X.shape[0]):
        # shift the image
        X_2d[i] = np.roll(X_2d[i], (offsets[i][0], offsets[i][1]), axis=(0,1))

    return np.reshape(X_2d, X.shape)

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width

"""
Method Author: Jon Barker
"""
def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size


    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors

def create_similar_duplicates(X, labels):
    """
    Aims to artificially create new data by taking
    each letter in X and replicating it with a slight
    modification. Replicates each letter 4 times by
    shifting them 1 pixels in each direction.

    Params:
    data - feature vector matrix
    labels - labels of the feature vectors in the matrix
    """
    X_2d = np.reshape(X, (X.shape[0],) + ORIGINAL_SAMPLE_DIMENSIONS_2D)

    shifted_right = np.roll(X_2d, (0,1), axis=(1,2))
    shifted_down = np.roll(X_2d, (1,0), axis=(1,2))

    shifted_left = np.roll(X_2d, (0,-1), axis=(1,2))
    shifted_up = np.roll(X_2d, (-1,0), axis=(1,2))

    shifted = np.concatenate((shifted_right,shifted_down,shifted_left,shifted_up))
    new_labels = np.concatenate((labels,labels,labels,labels))

    return (np.reshape(shifted, (shifted.shape[0], X.shape[1])), new_labels)

def artificially_increase_trainingset_size(fvectors_train_full, labels_train):
    """
    Aims to artificially increase the data set size by reproducing slight variations
    of each letter. Only reproduces those letteres which do not occur very often.
    Returns a data matrix with the new replicates.

    Params:
    data - feature vector matrix
    labels - labels of the feature vectors in the matrix
    """
    MAJORITY_CLASSES = ['e','t','a','o','h','i','n','s','r','l','d']
    minority_indices = np.argwhere(np.invert(np.isin(labels_train, MAJORITY_CLASSES))).flatten()

    duplicates, labels = create_similar_duplicates(fvectors_train_full[minority_indices,:], labels_train[minority_indices])

    fvectors_train_full = np.concatenate((fvectors_train_full, duplicates))
    labels_train = np.concatenate((labels_train, labels))

    return fvectors_train_full, labels_train

def increase_training_size(fvectors_train_full, labels_train, model_data):
    """
    Increases the given data matrix size by appending the training/validation data
    at the end of it

    Params:
    fvectors_train_full - feature vector matrix
    labels_train - labels of the feature vectors in the matrix
    model_data - dictionary storing data passed from training stage
    """
    testset='dev'

    # Construct a list of all the test set pages.
    page_names = ['data/{}/page.{}'.format(testset, page_num)
                  for page_num in range(1, 7)]

    # Load the correct labels for each test page
    true_labels = [utils.load_labels(page_name)
                   for page_name in page_names]

    # Load the 10-dimensional feature data for each test page
    page_data_all_pages = [load_page(page_name, model_data)
                           for page_name in page_names]

    for page, labels in zip(page_data_all_pages, true_labels):
        fvectors_train_full = np.concatenate((fvectors_train_full, page))
        labels_train = np.concatenate((labels_train, labels))

    return fvectors_train_full, labels_train

# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['bbox_size'] = bbox_size

    word_lists = [get_word_lists(filename) for filename in WORD_FILE_NAMES]
    model_data['word_lists'] = word_lists

    try:
        fvectors_train_full, labels_train = increase_training_size(fvectors_train_full, labels_train, model_data)
    except:
        print("Failed to increase training set size. Will proceed with base training set size.")

    print('Reducing to 10 dimensions')

    #fvectors_train_full, labels_train = artificially_increase_trainingset_size(fvectors_train_full,labels_train)


    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    model_data['labels_train'] = labels_train.tolist()

    return model_data

"""
Method Author: Jon Barker
"""
def load_page(page_name, model):
    """Load raw test data page.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    return fvectors_test

"""
Method Author: Jon Barker
"""
def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """ Applies k-nearest-neighbour to the given data and returns its predictions

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])


    predictions = kNearestNClassify(fvectors_train, labels_train, page)

    #model['fvectors_train'] = np.concatenate((fvectors_train, page))
    #model['labels_train'] = np.concatenate((labels_train, predictions))

    return predictions


def kNearestNClassify(train, train_labels, test, k=NEAREST_NEIGHBOURS):
    """k-Nearest neighbour classification.

    train - data matrix storing training data, one sample per row
    train_label - a vector storing the training data labels
    test - data matrix storing the test data
    k - the parameter k for the k-nearest neighbour algorithm

    returns: predictions
    """

    # Super compact implementation of k-nearest neighbour
    x= np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance


    nearest_k = np.argpartition(dist, -k, axis=1)[:,-k:]

    labels = train_labels[nearest_k]

    label = mode(labels, axis=1)[0].flatten()

    return label


def correct_errors(page, labels, bboxes, model):
    """
    Given a long list of character consecutive character predictions
    from a page, infers which characters belong to the same word by looking
    at the bounding boxes
    and corrects words that can not be found in a word dictionary and only
    need to be changed by 1 character to become a valid word.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """

    if not CORRECT_LABELS:
        return labels

    old_labels = np.copy(labels)
    new_word_indices = get_newword_indices(bboxes)

    #word_lists = [get_word_lists(filename) for filename in WORD_FILE_NAMES]
    word_lists = model['word_lists']

    word = []
    for i in range(len(labels)):
        if i in new_word_indices:
            matches = get_matches(''.join(word), word_lists)
            labels = replace_label(labels, word, matches, i)
            word = []

        word.append(labels[i].lower())

    return labels

def get_newword_indices(bboxes):
    """ Returns a 1d list of indices where each index is of a letter that starts
        a new word or is punctuation.

    parameters:

    bboxes - The bounding box coordinates of all the letters on the page.
    """
    dist_to_last_letter = get_distances_to_last_letter(bboxes)

    MAX_LETTER_DISTANCE_SAME_WORD = 6
    NEWLINE_INDICATOR = -20
    after_space_indices = np.argwhere(np.logical_or(\
                    dist_to_last_letter[:,0] > MAX_LETTER_DISTANCE_SAME_WORD, \
                    dist_to_last_letter[:,0] < NEWLINE_INDICATOR))

    character_sizes = get_character_sizes(bboxes)

    MAXIMUM_VERTICAL_STEPDOWN = -8
    PUNCTUATION_THRESHOLD = 15
    punctuation_indices = np.argwhere(np.logical_and(\
                    dist_to_last_letter[:,1] < MAXIMUM_VERTICAL_STEPDOWN, \
                    character_sizes[:,1] < PUNCTUATION_THRESHOLD))

    new_word_indices = np.concatenate((after_space_indices, punctuation_indices), axis=0)

    return new_word_indices

def replace_label(labels, word, matches, index):
    """
    Infers which characters belong to the same word from the bounding boxes
    and corrects words that can not be found in a word dictionary and only
    need to be changed by 1 character to become a valid word.

    parameters:

    labels - the full list of labels for the page
    word - the predicted word
    matches - a list of words that are similar to the predicted word
    index - the index+1 of the last character of the word inside the page labels list
    """
    if len(matches) == 0:
        return labels

    if word[-1] == ';' or 'â' in word:
        return labels

    # Simply takes the first matching word
    match = matches[0].lower()

    replace_index = None
    replace_char = None

    #print(word, match)
    counter = 0

    # finds the one character that needs to be corrected
    for a,b in zip(word,match):
        if a != b:
            replace_index = counter
            replace_char = b
            break
        counter += 1

    # finds the index in the page label list of that character
    label_index = index - len(word) + replace_index

    '''
    if true_labels[label_index] != replace_char:
        print(true_labels[label_index], ' predicted as ', labels[label_index], ' replaced by ', replace_char.lower())
        print(''.join(true_labels[index-len(word):index]), ' predicted as ', ''.join(word), ' replaced by ', match)
        print()
    '''

    labels[label_index] = replace_char.lower()

    return labels


def get_matches(word, word_lists):
    """
    Finds words similar to the given word if it is not a valid word.
    Similar words will differ by at most 1 character.

    parameters:

    word - a possible faulty word prediction
    word_lists - a list of dictionaries. Each dictionary contains lists of words indexed by word length
                 The word_list is ordered from most common to least common word dictionaries.
    """

    word_length = len(word)

    if word_length > 19:
        return []

    lowercase_word = word.lower()

    # Checks if the word matches any dictionary word
    for word_collections in word_lists:
        if lowercase_word in word_collections[str(word_length)]:
            return []

    matches = []
    MAXIMUM_DIFFERENCE_THRESHOLD = (word_length-1)/word_length
    if word_length != 1:
        MAXIMUM_DIFFERENCE_THRESHOLD -= 0.01

    for words in word_lists:
        matches = find_matches(lowercase_word, words, MAXIMUM_DIFFERENCE_THRESHOLD, word_length)

        if(len(matches) != 0):
            break

    return matches

def find_matches(word, words, threshold, word_length):
    """ From a list of strings finds the five strings that are most
        similar to the given word and are exactly equal apart from 1 character.

    parameters:

    word - A string whose similar words are needed
    words - A dictionary, indexed by word size, containing lists of words
    threshold - A value between 0 and 1 indicating the minimum similarity between
                two strings for them to be tagged as similar
    word_length - The length of the parameter 'word'
    """
    similar_words = difflib.get_close_matches(word, words[str(word_length)], n=5, cutoff=threshold)

    matches = [ match for match in similar_words \
                if different_by_one(word, match)]

    return matches

def different_by_one(word1, word2):
    """ Determines whether the two strings of equal size differ
        by at most 1 character with consideration of the order

    parameters:

    word1 - One of the two strings to compare with the other
    word2 - One of the two strings to compare with the other
    """

    mismatches = 0
    for a,b in zip(word1, word2):
        if a != b:
            mismatches += 1

            if mismatches == 2:
                return False

    return True




def get_word_lists(filename):
    """ Returns a dictionary of word lists of word-size N where
        the dictionary has N as the key. N ranges from 2 to 20

    parameters:

    bboxes - The filename to get the words from
    """
    with open(filename, 'r') as file:
        words = file.read().splitlines()

    word_lists = dict()
    word_lists[1] = []
    word_lengths = range(2,20)

    for i in word_lengths:
        words_group = [word.lower() for word in words if len(word) == i]
        word_lists[i] = words_group

    return word_lists

def get_character_sizes(bboxes):
    """ Returns an N*2 matrix where the two columns hold the width and height
        of a letter

    parameters:

    bboxes - The bounding box coordinates of all the letters on the page.
    """
    bottom_left_corners = bboxes[:, :2]
    top_right_corners = bboxes[:,2:]

    dimensions = np.subtract(top_right_corners, bottom_left_corners)

    return dimensions

def get_distances_to_last_letter(bboxes):
    """ Returns an N*2 matrix where the two columns hold the horizontal and the
        vertical distance of the letter from the last letter.

    parameters:

    bboxes - The bounding box coordinates of all the letters on the page.
    """
    ending_x_coordinates = bboxes[:-1,[2,3]]
    starting_x_coordinates = bboxes[1:, [0,3]]

    dist_to_last_letter = np.subtract(starting_x_coordinates, ending_x_coordinates)
    dist_to_last_letter = np.insert(np.subtract(starting_x_coordinates, ending_x_coordinates), 0, 0, axis=0)

    return dist_to_last_letter

def save_image_to_jpg(image, name):
    """
    Saves a PIL image to a jpg file
    """
    image = image.convert("L")
    image.save(name +".jpg", "JPEG")

def save_array_to_jpg(array, name, shape):
    """
    takes a numpy array representing an image and saves it as a jpg file
    """
    array = np.reshape(array, shape)
    try:
        imageio.imwrite(name+'.jpg', array)
    except:
        pass

def save_X_to_images(X, shape=ORIGINAL_SAMPLE_DIMENSIONS_2D):
    """
    saves every single sample in X as a seperate jpg file
    """

    counter = 0
    for i in range(len(X)):
        image = X[i,:]
        if counter > 1:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            im = plt.matshow(np.reshape(image, shape), cmap=cm.Greys_r);
            plt.colorbar(im)
            plt.show()
        if counter == 2:
            break
        save_array_to_jpg(image, str(i)+'Letter', shape)
        counter += 1
