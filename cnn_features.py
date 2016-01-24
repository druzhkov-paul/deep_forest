import caffe
import argparse
import cv2
from utils import load_annotation, print_progress_bar, stop_progress_bar
import numpy as np
from os.path import join, basename, splitext, isdir
from os import mkdir
import pickle as pkl


class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(formatter_class=ArgsFormatter,
                                 description="")
parser.add_argument("--annotation", required=True, help="")
parser.add_argument("--features_dir", required=True, help="")
parser.add_argument("--net_proto", required=True, help="")
parser.add_argument("--net_caffemodel", required=True, help="")
args = parser.parse_args()


def extract_features(annotation, image_size=(64, 64)):
    n = len(annotation)
    for i, a in enumerate(annotation):
        print_progress_bar(i, n)
        image_path = a["image"]
        label = a["label"]
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, image_size, image)
        image_channels = cv2.split(image)
        for channel_idx, channel in enumerate(image_channels):
            np.copyto(net.blobs["data"].data[0, channel_idx, :, :], channel)

        # image = np.dstack(cv2.split(image))
        # np.copyto(net.blobs["data"].data, image)
        # net.blobs["data"].data = image
        output_blobs = net.forward(end="conv1", blobs=["conv1", ])
        channels_num = output_blobs["conv1"].shape[1]
        channels = [output_blobs["conv1"][0, i, :, :] for i in range(channels_num)]
        features = cv2.merge(channels)
        output_dir = join(args.features_dir, "positives" if label else "negatives")
        if not isdir(output_dir):
            mkdir(output_dir)
        feature_map_path = join(output_dir, splitext(basename(image_path))[0] + ".pkl")
        pkl.dump(features, file(feature_map_path, "w"))
    stop_progress_bar()


image_size = (64, 64)
print("reading annotation...")
annotation = load_annotation(args.annotation)

caffe.set_mode_gpu()
net = caffe.Net(args.net_proto, args.net_caffemodel, caffe.TEST)
net.blobs["data"].reshape(1, 3, image_size[0], image_size[1])
net.reshape()
extract_features(annotation, image_size=image_size)
