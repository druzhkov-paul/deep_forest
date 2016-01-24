from rf_embedding import (RandomTreesEmbeddingSupervised, RandomTreesEmbeddingUnsupervised,
                          GBTreesEmbeddingSupervised, XGBTEmbeddingSupervised)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
# from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.base import clone
import sys
import argparse
import numpy as np
import cv2
import timeit
import pickle
# from os.path import join
from ast import literal_eval
from utils import load_annotation, print_progress_bar, stop_progress_bar


class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass

parser = argparse.ArgumentParser(formatter_class=ArgsFormatter,
                                 description="")
parser.add_argument("--annotation_train", required=True, help="")
parser.add_argument("--annotation_test", required=True, help="")
parser.add_argument("--patch_size", default=16, type=int, help="")
parser.add_argument("--patch_stride", default=4, type=int, help="")
parser.add_argument("--patches_per_image", default=100, help="")
parser.add_argument("--ntrees", default=10, type=int, help="")
parser.add_argument("--depth", default=5, type=int, help="")
parser.add_argument("--levels", default=2, type=int, help="")
parser.add_argument("--out", default="/tmp", help="")
parser.add_argument("--fe", default="rf", help="")
parser.add_argument("--classifier", default="gbt", help="")
parser.add_argument("--onehot", action="store_true", help="")
parser.add_argument("--njobs", default=8, type=int, help="")
parser.add_argument("--init", default="patches", help="patches or hogs or load")
parser.add_argument("--image_size", default="32,32", help="initial patch size")
args = parser.parse_args()

print(" ".join(sys.argv))

print("reading train annotation...")
annotation_train = load_annotation(args.annotation_train)
print("reading test annotation...")
annotation_test = load_annotation(args.annotation_test)


def extract_patches_2d(image, patch_size, patch_stride=1,
                       max_patches=None, random_state=None, reshape=True):
    from sklearn.utils.validation import check_array, check_random_state
    from sklearn.feature_extraction.image import extract_patches, _compute_n_patches

    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size
    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")
    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]
    stride = patch_stride
    if type(patch_stride) is not int:
        assert len(stride) == 2
        stride = np.ones(len(image.shape), dtype=int)
        stride[:2] = patch_stride
    extracted_patches = extract_patches(image,
                                        patch_shape=(p_h, p_w, n_colors),
                                        extraction_step=stride)
    if reshape:
        extracted_patches = extracted_patches.reshape(extracted_patches.shape[0], extracted_patches.shape[1], -1)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(extracted_patches.shape[0], size=n_patches)
        j_s = rng.randint(extracted_patches.shape[1], size=n_patches)
        patches = extracted_patches[i_s, j_s, :]
    else:
        patches = extracted_patches

    return patches


def transform_feature_map(feature_maps, feature_extractor, patch_size, patch_stride):
    for i, fm in enumerate(feature_maps):
        patches = extract_patches_2d(fm, patch_size, patch_stride)
        dim = patches.shape
        patches = patches.reshape(dim[0] * dim[1], -1)
        feature_maps[i] = feature_extractor.transform(patches).reshape(dim[0], dim[1], -1)
    return feature_maps


def draw_random_samples(feature_maps, labels, patch_size, patch_stride, patches_per_feature_map):
    X = []
    Y = []
    for image, label in zip(feature_maps, labels):
        if type(patches_per_feature_map) == list:
            patches_num = patches_per_feature_map[label]
        else:
            patches_num = patches_per_feature_map
        patches = extract_patches_2d(image, patch_size, patch_stride, max_patches=patches_num)
        X.append(patches)
        Y += [label for _ in patches]
    X = np.concatenate(X, axis=0)
    Y = np.array(Y)
    return X, Y


def classify_image(fm, label, receptive_size, stride, classifier):
    patches = extract_patches_2d(fm, receptive_size, stride)
    patches = patches.reshape(patches.shape[0] * patches.shape[1], -1)
    predictions = classifier.predict(patches)
    unique, counts = np.unique(predictions, return_counts=True)
    label_predicted = unique[np.argmax(counts)]
    return label, label_predicted


def evaluate(feature_maps, labels, receptive_size, stride, classifier, dump_file=None):
    from joblib import Parallel, delayed, cpu_count
    from sklearn.metrics import confusion_matrix

    start_time = timeit.default_timer()
    # njobs = cpu_count()
    d = Parallel(n_jobs=1)(delayed(classify_image)(fm, label, receptive_size, stride, classifier)
                           for fm, label in zip(feature_maps, labels))

    if dump_file is not None:
        with file(dump_file, "w") as f:
            for x in zip(*d):
                f.write(",".join(map(str, x)) + "\n")

    cm = confusion_matrix(*zip(*d))
    accuracy = float(cm.diagonal().sum()) / cm.sum()
    print("evaluation", timeit.default_timer() - start_time)
    return accuracy, cm


def patches_extractor(image):
    stride = (args.patch_stride, args.patch_stride)
    patches = extract_patches_2d(image, [args.patch_size, args.patch_size], patch_stride=stride)
    return patches


def hog_extractor(image):
    win_stride = (args.patch_stride, args.patch_stride)
    win_size = (args.patch_size, args.patch_size)
    hog_descriptor = cv2.HOGDescriptor(_winSize=win_size, _blockSize=(8, 8), _blockStride=(4, 4),
                                       _cellSize=(4, 4), _nbins=9)
    features = hog_descriptor.compute(image, winStride=win_stride)
    new_shape = ((image.shape[0] - win_size[0]) / win_stride[0] + 1,
                 (image.shape[1] - win_size[1]) / win_stride[1] + 1,
                 hog_descriptor.getDescriptorSize())
    features = features.reshape(new_shape)
    return features


# TODO. Add low level CNN features?
if args.init == "patches":
    initial_feature_extractor = patches_extractor
elif args.init == "hogs":
    initial_feature_extractor = hog_extractor
elif args.init != "load":
    print("Error. Unknown initial representation '{}'.".format(args.init))
    exit(0)


def get_initial_feature_representation(annotation):
    images = []
    labels = []
    image_size = literal_eval(args.image_size)
    start_time = timeit.default_timer()
    for a in annotation:
        image_path = a["image"]
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, image_size, image)
        feature_map = initial_feature_extractor(image)
        images.append(feature_map)
        labels.append(a["label"])
    print("initial", timeit.default_timer() - start_time)
    return images, labels

print("obtaining initial feature representation...")
if args.init == "load":
    images_train = pickle.load(file("images_train.pkl", "r"))
    labels_train = pickle.load(file("labels_train.pkl", "r"))
    images_test = pickle.load(file("images_test.pkl", "r"))
    labels_test = pickle.load(file("labels_test.pkl", "r"))
else:
    images_train, labels_train = get_initial_feature_representation(annotation_train)
    pickle.dump(images_train, file("images_train_" + args.init + ".pkl", "w"))
    pickle.dump(labels_train, file("labels_train_" + args.init + ".pkl", "w"))
    images_test, labels_test = get_initial_feature_representation(annotation_test)
    pickle.dump(images_test, file("images_test_" + args.init + ".pkl", "w"))
    pickle.dump(labels_test, file("labels_test_" + args.init + ".pkl", "w"))

base_classifier_ = None
if base_classifier_ is None:
    if args.classifier == "rf":
        base_classifier = RandomForestClassifier(n_estimators=1000, max_depth=15, n_jobs=args.njobs)
    elif args.classifier == "gbt":
        base_classifier = XGBClassifier(n_estimators=1500, max_depth=10, learning_rate=0.01)
    elif args.classifier == "linsvm":
        base_classifier = LinearSVC()
    else:
        print("Unknown classifier {}.".format(args.classifier))
        exit(0)
else:
    base_classifier = clone(base_classifier_)
print("=" * 80)
print("test: {} levels, {} trees, {} depth".format(args.levels, args.ntrees, args.depth))

assert args.levels >= 0
feature_extractors = []

subpatch_size = [2, 2]
subpatch_stride = [1, 1]
patches_per_image = map(int, args.patches_per_image.split(','))
if len(patches_per_image) == 1:
    patches_per_image = patches_per_image[0]

X = []
Y = []
for level in range(args.levels):
    print("extracting patches for training...")
    X, Y = draw_random_samples(images_train, labels_train, subpatch_size,
                               subpatch_stride, patches_per_image)
    print X.shape, Y.shape, np.mean(Y)

    print("training classifier...")
    start_time = timeit.default_timer()
    classifier = clone(base_classifier)
    classifier.fit(X, Y)
    print("train score", classifier.score(X, Y))
    print("training", timeit.default_timer() - start_time)

    print("evaluating classifier...")
    accuracy_train, cm_train = evaluate(images_train, labels_train, subpatch_size, subpatch_stride, classifier)
    print("Train accuracy: {:.2%}".format(accuracy_train))
    print(cm_train)
    print(cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis])
    accuracy_test, cm_test = evaluate(images_test, labels_test,
                                      subpatch_size, subpatch_stride, classifier,
                                      dump_file="test_predictions_{}.csv".format(level))
    print("Test accuracy: {:.2%}".format(accuracy_test))
    print(cm_test)
    print(cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis])

    print("\tgoing deeper to level {}...".format(level))
    print("training feature representation...")
    if args.fe == "rfu":
        feature_extractor = RandomTreesEmbeddingUnsupervised(n_estimators=args.ntrees,
                                                             max_depth=args.depth,
                                                             use_one_hot=args.onehot,
                                                             n_jobs=args.njobs)
        feature_extractor.fit(X)
        feature_extractors.append(feature_extractor)
    elif args.fe == "rfs":
        feature_extractor = RandomTreesEmbeddingSupervised(n_estimators=args.ntrees,
                                                           max_depth=args.depth,
                                                           use_one_hot=args.onehot,
                                                           n_jobs=args.njobs)
        feature_extractor.fit_transform(X, Y)
        feature_extractors.append(feature_extractor)
    elif args.fe == "gbt":
        feature_extractor = GBTreesEmbeddingSupervised(n_estimators=args.ntrees,
                                                       max_depth=args.depth,
                                                       use_one_hot=args.onehot,
                                                       n_jobs=args.njobs)
        feature_extractor.fit_transform(X, Y)
        feature_extractors.append(feature_extractor)
    elif args.fe == "xgbt":
        # For multiclass problems number of trees will be K times bigger.
        feature_extractor = XGBTEmbeddingSupervised(n_estimators=args.ntrees,
                                                    max_depth=args.depth,
                                                    use_one_hot=args.onehot,
                                                    silent=True)
        feature_extractor.fit_transform(X, Y)
        feature_extractors.append(feature_extractor)
    else:
        print("Error. Unknown feature extractor '{}'.".format(args.fe))
        exit(0)

    print("transforming train samples...")
    images_train = transform_feature_map(images_train, feature_extractor, subpatch_size, subpatch_stride)
    print("transforming test samples...")
    images_test = transform_feature_map(images_test, feature_extractor, subpatch_size, subpatch_stride)


print("extracting patches for training...")
X, Y = draw_random_samples(images_train, labels_train, subpatch_size, subpatch_stride, patches_per_image)
print X.shape, Y.shape
print("training final classifier...")
classifier = clone(base_classifier)
classifier.fit(X, Y)
print("evaluating classifier...")
accuracy_train, cm_train = evaluate(images_train, labels_train, subpatch_size, subpatch_stride, classifier)
print("Train accuracy: {:.2%}".format(accuracy_train))
print(cm_train)
print(cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis])
accuracy_test, cm_test = evaluate(images_test, labels_test, subpatch_size,
                                  subpatch_stride, classifier,
                                  dump_file="test_predictions_{}.csv".format(args.levels))
print("Test accuracy: {:.2%}".format(accuracy_test))
print(cm_test)
print(cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis])


# ntrees = np.fromstring(args.ntrees, sep=",", dtype=int)
# depths = np.fromstring(args.depth, sep=",", dtype=int)
# for trees in ntrees:
#     for depth in depths:
#         test(args.levels, ntrees=trees, depth=depth,
#              base_classifier_=RandomForestClassifier(n_estimators=100, max_depth=5))
