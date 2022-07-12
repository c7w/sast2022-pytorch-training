import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


def calc_label(label: np.ndarray, threshold: float):
    """
    Calc label category statistics.
    :param label: A numpy array, shaped (H, W).
    :param threshold: float number.
    :return: {"mountain": bool, "sky": bool, "water": bool, "human": bool}
    """

    label2id = {
        "mountain": [0, 7],
        "sky": [1],
        "water": [2, 3, 8, 16, 20],
    }

    return {
        k: np.isin(label, v).sum() > (np.ones_like(label).sum() * threshold) for k, v in label2id.items()
    }


def process_data(mode: str, threshold: float):
    """
    Pre-process data.
    :param mode: Either in `train`, `val` or `test`
    :param threshold: threshold to determine a category.
    :return: None. Write a file to the corresponding path.
    """
    working_dir = (Path(__file__) / ".." / ".." / "data" / mode).resolve()
    image_dir = working_dir / "imgs"
    label_dir = working_dir / "labels"
    print(f"[Data] Now in {working_dir}...")

    out_str = "img_path,mountain,sky,water\n"

    assert os.path.exists(image_dir), "No directory called `imgs` found in working directory!"
    assert os.path.exists(label_dir), "No directory called `labels` found in working " \
                                                                "directory!"

    filename_list = [x[:-4] for x in os.listdir(image_dir)]
    for idx, file_name in tqdm(enumerate(filename_list), total=len(filename_list)):
        label_path = str(label_dir / f"{file_name}.png")
        label = Image.open(label_path)
        label_array = np.array(label)

        statistics = calc_label(label_array, threshold)
        out_str += f"{file_name}.jpg,{statistics['mountain']},{statistics['sky']},{statistics['water']}\n"

        # if idx == 1000:
        #     break

    # After all file has been processed, write `out_str` to `{working_dir}/file.txt`
    with open(f"{working_dir}/file.txt", 'w+') as file:
        file.write(out_str)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for determining if a label exists in "
                                                                   "the image.")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    process_data(args.mode, args.threshold)
