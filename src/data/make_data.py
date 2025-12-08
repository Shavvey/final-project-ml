from typing import Optional
import numpy as np
from numpy.typing import NDArray
from torchvision import transforms
import data.dataframe as df
from data.dataset import ACRIMA


def get_labels_and_images() -> tuple[NDArray, NDArray]:
    """Collects all images and labels into two numpy arrays"""
    dfs = df.DataFrame.collect(df.DATA_DIRECTORY)
    labels = df.DataFrame.get_labels(dfs)
    images = df.DataFrame.get_images(dfs)
    return images, labels


def get_ACRIMA(
    transform: Optional[transforms.Compose] = None,
    from_serialized_npz: Optional[bool] = None,
) -> ACRIMA:
    """Get `Dataset` class for ACRIMA, optionally apply a transform to its elements"""
    if from_serialized_npz == None:
        images, labels = get_labels_and_images()
        if transform == None:
            return ACRIMA(images, labels)
        else:
            return ACRIMA(images, labels, transform)
    else:
        # TODO: implement npz compression and uncompression
        # this will improve IO performance
        raise Exception("Not implemented yet!")


def serialize_into_npz_arrays(npz_array_path: str):
    """ Serialize data into npz array (compressed numpy array).
    Main motivation behind this is to improve IO performance """
    FILEPATH: str = npz_array_path + "ACRIMA.npz"
    images, labels = get_labels_and_images()
    np.savez_compressed(FILEPATH, images=images, labels=labels)
