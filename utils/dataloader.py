import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle as pickle

from sklearn.preprocessing import OrdinalEncoder
import os

from joblib import Parallel, delayed
from joblib import parallel_backend
import torch
from torch.nn.utils.rnn import pad_sequence


class sp_loc_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_root,
        user=None,
        dataset="geolife",
        data_type="train",
        previous_day=7,
        model_type="transformer",
        day_selection="default",
        inductive_pct=0,
        get_testinductive=False,
        exp_num=0,
    ):
        self.root = source_root
        self.user = user
        self.data_type = data_type
        self.previous_day = previous_day
        self.model_type = model_type
        self.dataset = dataset
        self.day_selection = day_selection
        self.inductive_pct = inductive_pct
        self.get_testinductive = get_testinductive

        # check whether to train individual models
        if user is None:
            self.is_individual_model = False
        else:
            self.is_individual_model = True

        # define data storing dir
        self.data_dir = os.path.join(source_root, dataset)
        if day_selection == "default":
            if self.inductive_pct > 0:
                
                if data_type == 'test':
                    save_path = os.path.join(
                        self.data_dir,
                        f"{self.dataset}_{previous_day}_{data_type}.pk",
                )
                else:
                    save_path = os.path.join(
                        self.data_dir,
                        f"{self.dataset}_{previous_day}_{data_type}_remove{self.inductive_pct}pct_{exp_num}.pk",
                    )
                if self.get_testinductive:
                    save_path = os.path.join(
                        self.data_dir,
                        f"{self.dataset}_{previous_day}_testinductive{self.inductive_pct}pct_{exp_num}.pk",
                    )

            else:
                save_path = os.path.join(
                    self.data_dir,
                    # f"{self.dataset}_{self.model_type}_{previous_day}_{data_type}.pk",
                    f"{self.dataset}_{previous_day}_{data_type}.pk",
                )
        else:
            save_path = os.path.join(
                self.data_dir,
                f"{self.dataset}_{''.join(str(x) for x in day_selection)}_{data_type}.pk",
            )

        # if the file is pre-generated we load the file, otherwise run self.generate_data()
        if Path(save_path).is_file():
            self.data = pickle.load(open(save_path, "rb"))
        else:
            print("No dataset found; generating dataset...")
            parent = Path(save_path).parent.absolute()
            if not os.path.exists(parent):
                os.makedirs(parent)
            self.data = self.generate_data()

        self.len = len(self.data)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        """Get a single sample."""
        selected = self.data[idx]

        return_dict = {}
        # [sequence_len]
        x = torch.tensor(selected["X"])
        # [1]
        y = torch.tensor(selected["Y"])

        # [1]
        return_dict["user"] = torch.tensor(selected["user_X"])  # [1]
        # [sequence_len] in half an hour
        return_dict["time"] = torch.tensor(selected["start_min_X"] // 30)
        return_dict["start_min_X"] = torch.tensor(selected["start_min_X"])
        return_dict["start_min_Y"] = torch.tensor(selected["start_min_Y"])  # [1]
        
        if self.dataset == "geolife":
            return_dict["dur_X"] = torch.tensor(selected["dur_X"] // 30, dtype=torch.long)
        
        #
        return_dict["diff"] = torch.tensor(selected["diff"])

        # [sequence_len]
        return_dict["weekday_X"] = torch.tensor(selected["weekday_X"], dtype=torch.int64)
        return_dict["weekday_Y"] = torch.tensor(selected["weekday_Y"], dtype=torch.int64)  # [1]

        # add coordinate - lat long
        return_dict["lat_X"] = torch.tensor(selected["lat_X"], dtype=torch.float64)
        return_dict["lon_X"] = torch.tensor(selected["lon_X"], dtype=torch.float64)
        return_dict["x_X"] = torch.tensor(selected["x_X"], dtype=torch.float64)
        return_dict["y_X"] = torch.tensor(selected["y_X"], dtype=torch.float64)

        return x, y, return_dict

    def generate_data(self):
        # the valid location ids for unifying comparision
        if self.dataset == "geolife":
            self.valid_ids = pd.read_csv(os.path.join(self.data_dir, f"valid_ids_{self.dataset}.csv"))["id"].values
        else:
            self.valid_ids = load_pk_file(
                os.path.join(self.data_dir, f"valid_ids_{self.dataset}.pk")
            )

        # the location data
        ori_data = pd.read_csv(os.path.join(self.data_dir, f"dataSet_{self.dataset}.csv"))
        ori_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

        # truncate too long duration: > 2days to 2 days
        if self.dataset == "geolife":
            ori_data.loc[ori_data["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1

        if self.model_type == "mobtcast":
            # get the location file and obtain their center
            loc_file = pd.read_csv(
                os.path.join(self.root, f"locations_{self.dataset}.csv")
            )

            # loc_file = gpd.GeoDataFrame(loc_file, crs="EPSG:4326", geometry="center")
            # loc_file["lng"] = loc_file.geometry.x
            # loc_file["lat"] = loc_file.geometry.y
            loc_file.rename(
                columns={"longitude": "lng", "latitude": "lat"}, inplace=True
            )
            # loc_file.drop(columns=["center", "user_id", "extent"], inplace=True)

            # merge sp with loc geom
            ori_data = ori_data.merge(loc_file, left_on="location_id", right_on="id")

            ori_data.rename(columns={"id_x": "id"}, inplace=True)
            ori_data.drop(columns={"id_y"}, inplace=True)

        # classify the datasets, user dependent 0.6, 0.2, 0.2
        train_data, vali_data, test_data = self._splitDataset(ori_data)

        # encode unseen location in train as 1 (0 reserved for padding)
        # this saves (a bit of) #parameters when defining the model
        train_data, vali_data, test_data, enc = self._encode_loc(
            train_data, vali_data, test_data
        )

        if self.model_type == "mobtcast":
            # re encode location, use the enc from self._encode_loc()
            loc_file["id"] = enc.transform(loc_file["id"].values.reshape(-1, 1)) + 2

            # filter and transform to list
            loc_file = loc_file.loc[loc_file["id"] != 1].sort_values(by="id")
            loc_file = loc_file[["lng", "lat"]].values.tolist()
            # save for training
            save_path = os.path.join(
                self.data_dir, f"{self.dataset}_loc_{self.previous_day}.pk"
            )
            save_pk_file(save_path, loc_file)
        print(
            f"Max location id:{train_data.location_id_num.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
        )

        # Output train, valid, test data
        train_save_path = os.path.join(self.data_dir, f"{self.dataset}_train.csv")
        valid_save_path = os.path.join(self.data_dir, f"{self.dataset}_valid.csv")
        test_save_path = os.path.join(self.data_dir, f"{self.dataset}_test.csv")
        train_data.to_csv(train_save_path, index=False)
        vali_data.to_csv(valid_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)

        # preprocess the data into sequences
        train_records = self._preProcessDatasets(train_data, "train")
        validation_records = self._preProcessDatasets(vali_data, "validation")
        test_records = self._preProcessDatasets(test_data, "test")

        if self.data_type == "test":
            return test_records
        if self.data_type == "validation":
            return validation_records
        if self.data_type == "train":
            return train_records

    def _encode_loc(self, train, validation, test):
        """encode unseen locations in validation and test into 1 (0 reserved for padding)."""
        # fit to train
        enc = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
        ).fit(train["location_id"].values.reshape(-1, 1))

        # Output the encoder categories
        categories = pd.DataFrame(enc.categories_[0], columns=["location_id"])
        # location_id_num is the index + 2
        categories["location_id_num"] = np.arange(len(categories)) + 2
        categories.to_csv(
            os.path.join(self.data_dir, f"{self.dataset}_location_id.csv"), index=False
        )

        # apply to all. add 2 to account for unseen locations (1) and to account for 0 padding
        train["location_id_num"] = (
            enc.transform(train["location_id"].values.reshape(-1, 1)) + 2
        )
        validation["location_id_num"] = (
            enc.transform(validation["location_id"].values.reshape(-1, 1)) + 2
        )
        test["location_id_num"] = (
            enc.transform(test["location_id"].values.reshape(-1, 1)) + 2
        )

        return train, validation, test, enc

    def _splitDataset(self, totalData):
        """Split dataset into train, vali and test."""
        totalData = totalData.groupby("user_id", group_keys=False).apply(
            self.__getSplitDaysUser
        )

        train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
        vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
        test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

        # final cleaning
        train_data.drop(columns={"Dataset"}, inplace=True)
        vali_data.drop(columns={"Dataset"}, inplace=True)
        test_data.drop(columns={"Dataset"}, inplace=True)

        return train_data, vali_data, test_data

    def __getSplitDaysUser(self, df):
        """Split the dataset according to the tracked day of each user."""
        maxDay = df["start_day"].max()
        train_split = maxDay * 0.6
        vali_split = maxDay * 0.8

        df["Dataset"] = "test"
        df.loc[df["start_day"] < train_split, "Dataset"] = "train"
        df.loc[
            (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
            "Dataset",
        ] = "vali"

        return df

    def _preProcessDatasets(self, data, dataset_type):
        """Generate the datasets and save to the disk."""
        valid_records = self.__getValidSequence(data)

        valid_records = [item for sublist in valid_records for item in sublist]

        if self.day_selection == "default":
            save_path = os.path.join(
                self.data_dir,
                # f"{self.dataset}_{self.model_type}_{self.previous_day}_{dataset_type}.pk",
                f"{self.dataset}_{self.previous_day}_{dataset_type}.pk",
            )
        else:
            save_path = os.path.join(
                self.data_dir,
                f"{self.dataset}_{''.join(str(x) for x in self.day_selection)}_{dataset_type}.pk",
            )

        save_pk_file(save_path, valid_records)

        return valid_records

    def __getValidSequence(self, input_df):
        """Get the valid sequences.

        According to the input previous_day and day_selection.
        The length of the history sequence should >2, i.e., whole sequence >3.

        We use parallel computing on users (generation is independet of users) to speed up the process.
        """
        valid_user_ls = applyParallel(
            input_df.groupby("user_id"), self.___getValidSequenceUser, n_jobs=-1
        )
        return valid_user_ls

    def ___getValidSequenceUser(self, df):
        """Get the valid sequences per user.

        input df contains location history for a single user.
        """

        df.reset_index(drop=True, inplace=True)

        data_single_user = []

        # get the day of tracking
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records that do not include enough previous_day
            if row["diff_day"] < self.previous_day:
                continue

            # get the history records [curr-previous, curr]
            hist = df.iloc[:index]
            hist = hist.loc[
                (hist["start_day"] >= (row["start_day"] - self.previous_day))
            ]

            # should be in the valid user ids
            if not (row["id"] in self.valid_ids):
                continue

            if self.day_selection != "default":
                # get only records from selected days
                hist["diff"] = row["diff_day"] - hist["diff_day"]
                hist = hist.loc[hist["diff"].isin(self.day_selection)]
                if len(hist) < 2:
                    continue

            data_dict = {}

            # get the features: location, user, weekday, start time, duration, diff to curr day, and poi
            data_dict["X"] = hist["location_id_num"].values
            data_dict["lon_X"] = hist["longitude"].values
            data_dict["lat_X"] = hist["latitude"].values
            data_dict["x_X"] = hist["x"].values
            data_dict["y_X"] = hist["y"].values

            try:
                data_dict["user_X"] = hist["user_id"].values[0]
            except:
                print(hist)
            data_dict["weekday_X"] = hist["weekday"].values
            data_dict["start_min_X"] = hist["start_min"].values
            data_dict["timestamp_X"] = hist["timestamp"].values
            if self.dataset == "geolife":
                data_dict["dur_X"] = hist["duration"].astype(int).values
            data_dict["diff"] = (row["diff_day"] - hist["diff_day"]).astype(int).values

            # the next location is the target
            data_dict["Y"] = int(row["location_id_num"])
            data_dict["lon_Y"] = row["longitude"]
            data_dict["lat_Y"] = row["latitude"]
            data_dict["x_Y"] = row["x"]
            data_dict["y_Y"] = row["y"]

            # Also output the time for the next location
            data_dict["weekday_Y"] = int(row["weekday"])
            data_dict["start_min_Y"] = int(row["start_min"])
            data_dict["timestamp_Y"] = row["timestamp"]

            # Also output the category name of the POI
            if self.dataset == "geolife" or self.dataset == "gowalla_london":
                pass
            else:
                data_dict["poi_cname_X"] = hist["name"].values

            # append the single sample to list
            data_single_user.append(data_dict)

        return data_single_user


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pk_file(save_path):
    """Function to load data from pickle format given path."""
    return pickle.load(open(save_path, "rb"))


def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    """
    Funtion warpper to parallelize funtions after .groupby().
    Parameters
    ----------
    dfGrouped: pd.DataFrameGroupBy
        The groupby object after calling df.groupby(COLUMN).
    func: function
        Function to apply to the dfGrouped object, i.e., dfGrouped.apply(func).
    n_jobs: int
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description
    print_progress: boolean
        If set to True print the progress of apply.
    **kwargs:
        Other arguments passed to func.
    Returns
    -------
    pd.DataFrame:
        The result of dfGrouped.apply(func)
    """
    with parallel_backend("threading", n_jobs=n_jobs):
        df_ls = Parallel()(
            delayed(func)(group, **kwargs)
            for _, group in tqdm(dfGrouped, disable=not print_progress)
        )
    return df_ls


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    x_batch, y_batch = [], []

    # get one sample batch
    x_dict_batch = {"len": []}
    for key in batch[0][-1]:
        x_dict_batch[key] = []

    for src_sample, tgt_sample, return_dict in batch:
        x_batch.append(src_sample)
        y_batch.append(tgt_sample)

        x_dict_batch["len"].append(len(src_sample))
        for key in return_dict:
            x_dict_batch[key].append(return_dict[key])

    x_batch = pad_sequence(x_batch)  # does not specify batch_first=True, so 
    # the returned tensor will be (L, B)
    y_batch = torch.tensor(y_batch, dtype=torch.int64)

    # x_dict_batch
    x_dict_batch["user"] = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
    x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
    x_dict_batch["start_min_Y"] = torch.tensor(x_dict_batch["start_min_Y"], dtype=torch.int64)
    x_dict_batch["weekday_Y"] = torch.tensor(x_dict_batch["weekday_Y"], dtype=torch.int64)
    for key in x_dict_batch:
        if key in ["user", "len", "history_count", "start_min_Y", "weekday_Y"]:
            continue
        x_dict_batch[key] = pad_sequence(x_dict_batch[key])

    return x_batch, y_batch, x_dict_batch


def test_dataloader(train_loader):
    batch_size = train_loader.batch_size
    # print(batch_size)

    x_shape = 0
    x_dict_shape = 0
    for batch_idx, (x, y, x_dict) in tqdm(enumerate(train_loader)):
        # print("batch_idx ", batch_idx)
        x_shape += x.shape[0]
        x_dict_shape += x_dict["duration"].shape[0]
        # print(x_dict["user"].shape)
        # print(x_dict["poi"].shape)

        # print(, batch_len)

        # print(data)
        # print(target)
        # print(dict)
        # if batch_idx > 10:
        #     break
    print(x_shape / len(train_loader))
    print(x_dict_shape / len(train_loader))


if __name__ == "__main__":
    source_root = r"./data/"

    dataset_train = sp_loc_dataset(
        source_root, dataset="geolife", data_type="train", previous_day=7
    )
    kwds_train = {"shuffle": False, "num_workers": 0, "batch_size": 2}
    train_loader = torch.utils.data.DataLoader(
        dataset_train, collate_fn=collate_fn, **kwds_train
    )

    test_dataloader(train_loader)
