from dataclasses import dataclass
import base64 as b64
from pathlib import Path

DATA_DIRECTORY = "ACRIMA_data"


@dataclass(frozen=True)  # ensures dataframe is read only
class DataFrame:
    image: bytes  # bytes that represent the dataframe
    g_label: bool  # either gluacoma (1) or not gluaomcoma (0)

    @staticmethod
    def _df_from_image(image_path: Path) -> "DataFrame":
        """"""
        # get label, if _g_ is in filename. label is true, else false
        filename: str = image_path.name
        g_label = True if filename.__contains__("_g_") else False
        with open(image_path, "rb") as image:
            # read image as a series of bytes
            image_data = image.read()
            enc_bytes = b64.b64encode(image_data)
            return DataFrame(enc_bytes, g_label)

    @staticmethod
    def collect(data_dir: str) -> list["DataFrame"]:
        """ Collect images inside directory into a list of dataframes """
        data_dir_path = Path(data_dir)
        dfs: list[DataFrame] = []
        # walk data directory, works similar to os.walk()
        for root, _, files in data_dir_path.walk(follow_symlinks=False):
            for file in files:
                path = root / file # append filename to root path to get path to image
                df = DataFrame._df_from_image(path)
                dfs.append(df)
        return dfs

    @staticmethod
    def get_images(dfs: list["DataFrame"]) -> list[bytes]:
        return [df.image for df in dfs]

    @staticmethod
    def get_labels(dfs: list["DataFrame"]) -> list[bool]:
        return [df.g_label for df in dfs]
