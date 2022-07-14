import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


def calc_label(label: np.ndarray, threshold: float):
    """
    Calc label category statistics.
    For all label_ids in `label` array, calculate the total number of mountains (namely how many label_ids is in [0,7]),
    if this number is greater than threshold * sizeof label, then mark the `mountain` field in return dictionary as
    True, else as False.
    :param label: A numpy array, shaped (H, W).
    :param threshold: float number.
    :return: {"mountain": bool, "sky": bool, "water": bool}
    """

    label2id = {
        "mountain": [0, 7],
        "sky": [1],
        "water": [2, 3, 8, 16, 20],
    }

    # TODO Start: Finish this function #
    raise NotImplementedError
    return {"mountain": True, "sky": True, "water": True}
    # TODO End #


def process_data(mode: str, threshold: float):
    """
    Pre-process data.
    :param mode: Either in `train`, `val` or `test`
    :param threshold: threshold to determine a category.
    :return: None. Write a file to the corresponding path.
    """
    working_dir = (Path(__file__) / ".." / ".." / "data" / mode).resolve()

    # TODO Start: Append directory in pathlib.Path, so that they point to `./data/{mode}/imgs`
    #  and `./data/{mode}/labels` #
    image_dir = None
    label_dir = None
    # TODO End #

    print(f"[Data] Now in {working_dir}...")

    out_str = "img_path,mountain,sky,water\n"

    assert os.path.exists(image_dir), "No directory called `imgs` found in working directory!"
    assert os.path.exists(label_dir), "No directory called `labels` found in working " \
                                                                "directory!"

    # TODO Start: Construct a list of filenames without suffix from image_dir, like ['48432_b67ec6cd63_b',
    #  '70190_90b25efb3b_b', ...] #
    filename_list = [None, None, ...]
    # TODO End #

    for idx, file_name in tqdm(enumerate(filename_list), total=len(filename_list)):
        label_path = str(label_dir / f"{file_name}.png")
        label = Image.open(label_path)
        label_array = np.array(label)

        statistics = calc_label(label_array, threshold)
        out_str += f"{file_name}.jpg,{statistics['mountain']},{statistics['sky']},{statistics['water']}\n"

        # if idx == 1000:
        #     break

    # After all file has been processed, write `out_str` to `{working_dir}/file.txt`
    # TODO Start: Write out_str to `{working_dir}/file.txt` in overwritten mode #
    raise NotImplementedError
    # TODO End #


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for determining if a label exists in "
                                                                   "the image.")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    process_data(args.mode, args.threshold)
