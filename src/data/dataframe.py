from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np
import numpy.typing as npt

DATA_DIRECTORY = "ACRIMA_data"
RESIZE_DIMS = (350, 350)


@dataclass(frozen=True)  # ensures dataframe is read only
class DataFrame:
    image_array: npt.NDArray
    g_label: bool  # either gluacoma (1) or not gluaomcoma (0)
    filename: str

    @staticmethod
    def _df_from_image(image_path: Path) -> "DataFrame":
        """Internal method to load dataframe given an image path"""
        # get label, if _g_ is in filename. label is true, else false
        filename: str = image_path.name
        g_label = True if filename.__contains__("_g_") else False
        with Image.open(image_path) as image:
            image = image.resize(RESIZE_DIMS)
            image_array = np.asarray(image)
        return DataFrame(image_array, g_label, filename)

    @staticmethod
    def collect(data_dir: str) -> list["DataFrame"]:
        """Collect images inside directory into a list of dataframes"""
        data_dir_path = Path(data_dir)
        dfs: list[DataFrame] = []
        # walk data directory, works similar to os.walk()
        for root, _, files in data_dir_path.walk(follow_symlinks=False):
            for file in files:
                path = root / file  # append filename to root path to get path to image
                df = DataFrame._df_from_image(path)
                dfs.append(df)
        return dfs

    @staticmethod
    def get_images(dfs: list["DataFrame"]) -> npt.NDArray:
        image_list = []
        for df in dfs:
            image_list.append(df.image_array)
        return np.array(image_list)

    @staticmethod
    def get_labels(dfs: list["DataFrame"]) -> npt.NDArray:
        """Returns all the labels inside the dataframes"""
        convert = lambda x: 1 if x is True else -1
        labels = list(map(convert, [df for df in dfs]))
        return np.array(labels)
