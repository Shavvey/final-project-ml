from typing import Optional
import numpy as np
from numpy.typing import NDArray
import data.dataframe as df
from data.dataset import ACRIMA, Dataset

COMPRESSED_LABELS_FILE = "ACRIMA_data/serialized/labels.npz"
COMPRESSED_IMAGES_FILE = "ACRIMA_data/serialized/images_350.npz"


def get_labels_and_images() -> tuple[NDArray, NDArray]:
    dfs = df.DataFrame.collect(df.DATA_DIRECTORY)
    labels = df.DataFrame.get_labels(dfs)
    images  = df.DataFrame.get_images(dfs)
    return images, labels


def get_ACRIMA(from_serialized_npz: Optional[bool] = None) -> ACRIMA:
    if from_serialized_npz == None:
        images, labels = get_labels_and_images()
        return ACRIMA(images, labels)
    else:
        raise Exception("Not implemented yet!")
    
