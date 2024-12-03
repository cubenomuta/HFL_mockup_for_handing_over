import os
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
from torch.utils.data import Dataset

from logging import DEBUG, INFO, ERROR
from flwr.common.logger import log


class NIH_CXR(Dataset):
    """
    Custom dataset class for NIH-CXR data.
    Automatically selects training or testing data based on `train` flag.
    """

    def __init__(
            self,
            centralized: bool = False,
            train: bool = True,
            transform=None,
            target_transform=None
        ):
        """
        Args:
            train (bool): Trueなら学習データ、Falseならテストデータを使用。
            transform (callable, optional): 画像に適用する前処理。
            target_transform (callable, optional): ラベルに適用する前処理。
        """
        if centralized: # 全体のデータ数が多すぎるため
            if train:
                log(
                    ERROR,
                    "centralized train data is needed"
                )
            else:
                self.csv_file = Path("data/NIH_CXR/nih-cxr-lt_single-label_balanced-test.csv")
                self.data_root = Path("data/NIH_CXR/balanced_test_data")
        else:
            if train:
                self.csv_file = Path("data/NIH_CXR/nih-cxr-lt_single-label_train.csv")
                self.data_root = Path("data/NIH_CXR/train_data")
            else:
                self.csv_file = Path("data/NIH_CXR/nih-cxr-lt_single-label_test.csv")
                self.data_root = Path("data/NIH_CXR/test_data")

        self.transform = transform
        self.target_transform = target_transform

        self.data, self.targets = self.__load_data__()

    def __load_data__(self):
        """
        CSVファイルを読み込み、画像パスとラベルを生成。
        """
        try:
            df = pd.read_csv(self.csv_file)

            image_paths = [self.data_root / img_id for img_id in df["id"]]

            one_hot_labels = df.iloc[:, 1:-1].values  # `id`列と`subject_id`列を除外
            labels = np.argmax(one_hot_labels, axis=1)  # ワンホット→整数ラベル

            return image_paths, labels

        except FileNotFoundError as e:
            log(ERROR, f"CSV file not found: {self.csv_file} - {e}")
            raise
        except pd.errors.EmptyDataError as e:
            log(ERROR, f"CSV file is empty: {self.csv_file} - {e}")
            raise
        except Exception as e:
            log(ERROR, f"Unexpected error while loading data: {e}")
            raise

    def __getitem__(self, index):
        """
        指定されたインデックスのデータ（画像とラベル）を取得
        """
        try:
            # ファイルパスとラベルを取得
            img_path = self.data[index]
            label = self.targets[index]

            # ファイルパスを画像として開く
            image = Image.open(img_path).convert("RGB")
            label = torch.tensor(label, dtype=torch.long)

            # transformを適用
            if self.transform is not None:
                image = self.transform(image)

            if self.target_transform is not None:
                label = self.target_transform(label)

            return image, label

        except Exception as e:
            log(ERROR, f"Error in __getitem__ at index {index}: {e}")
            raise

    def __len__(self):
        """
        データセットのサイズを返す
        """
        return len(self.data)