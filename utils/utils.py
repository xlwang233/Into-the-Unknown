import yaml
import json
import random, torch, os, logging, json
import numpy as np
import pandas as pd
import torch.nn as nn
from easydict import EasyDict as edict
import pickle

from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP

from utils.train import train_net, test, get_performance_dict
from utils.dataloader import sp_loc_dataset, collate_fn

from models.MHSA import TransEncoder
from models.loc_embedder import LocCLIPLightning, GlobalPositionEncDec
from embed.ctle import CTLE, CTLEEmbedding, PositionalEncoding, TemporalEncoding, MaskedLM, MaskedHour
from embed.static import DownstreamEmbed, StaticEmbed
from embed.hier import HierEmbedding, Hier


def load_cllp_model(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    lightning_model = LocCLIPLightning(**ckpt["hyper_parameters"]).to(device)
    lightning_model.load_state_dict(ckpt["state_dict"])
    lightning_model.eval()

    model = lightning_model.model

    if return_all:
        return model
    else:
        return model.loc_enc


def load_embed_model(embed_name, save_dir, exp_num=None):

    if embed_name == "vanilla":
        return None
    
    ckpt_path = os.path.join(save_dir, f"checkpoint_{exp_num}.pt")
    config_path = os.path.join(save_dir, "conf.json")
    
    # Load the config json file
    with open(config_path, "r") as f:
        config = json.load(f)
    config = edict(config)
    if embed_name == "ctle":
        ckpt = torch.load(ckpt_path)
        if config["ctle_static"] == True:
            embed_mat = np.load(os.path.join(save_dir, f"embed_mat_{exp_num}.npy"))
            embed_layer = StaticEmbed(embed_mat)
            return embed_layer
        else:
            embed_model, _ = init_embed_models(config)
            embed_model.load_state_dict(ckpt)
            embed_model.eval()
            # embed_model.to(device)
            return embed_model    
    else:
        embed_mat = np.load(os.path.join(save_dir, f"embed_mat_{exp_num}.npy"))
        embed_layer = StaticEmbed(embed_mat)
        return embed_layer


def load_config(path):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trained_nets(config, model, train_loader, val_loader, test_loader, device, log_dir, logger):
    best_model, performance = train_net(config, model, train_loader, val_loader, test_loader,
                                        device, log_dir, logger)
    performance["type"] = "vali"

    return best_model, performance


def get_test_result(config, best_model, test_loader, device, logger):

    return_dict, result_arr_user, df_res = test(config, best_model, test_loader, device, logger)

    performance = get_performance_dict(return_dict)
    performance["type"] = "test"
    # print(performance)

    result_user_df = pd.DataFrame(result_arr_user).T
    result_user_df.columns = [
        "correct@1",
        "correct@3",
        "correct@5",
        "correct@10",
        "rr",
        "ndcg",
        "total",
    ]
    result_user_df.index.name = "user"

    return performance, result_user_df, df_res


def init_embed_models(config, logger=None):
    if config["embed_name"] == "ctle":
        encoding_layer = PositionalEncoding(config.embed_size, max_len=1000)
        if config["encoding_type"] == 'temporal':
            encoding_layer = TemporalEncoding(config["embed_size"])
        obj_models = [MaskedLM(config["embed_size"], config["total_loc_num"])]
        if config["ctle_objective"] == "mh":
            obj_models.append(MaskedHour(config["embed_size"]))
        obj_models = nn.ModuleList(obj_models)

        ctle_embedding = CTLEEmbedding(encoding_layer, config["embed_size"], config["total_loc_num"])
        embed_model = CTLE(ctle_embedding, config["hidden_size"], 
                          num_layers=config["ctle_num_layers"], 
                          num_heads=config["ctle_num_heads"],
                          init_param=config["init_param"], 
                          detach=config["ctle_detach"])
        
        total_params = sum(p.numel() for p in embed_model.parameters() if p.requires_grad)
        if logger is not None:
            logger.info(f"Total number of trainable parameters: {total_params}")
            logger.info(embed_model)

        return embed_model, obj_models
    elif config["embed_name"] == "hier":
        hier_embedding = HierEmbedding(config["token_embed_size"], config["total_loc_num"],
                                       config["week_embed_size"], config["hour_embed_size"], 
                                       config["duration_embed_size"])
        hier_model = Hier(hier_embedding, hidden_size=config["hidden_size"], 
                          num_layers=config["num_layers"], share=config["hier_share"])
        return hier_model, None


def get_models(config, device, logger, exp_num):
    config = edict(config)
    total_params = 0

    if config.loc_embed_method == "calliper":
        loc_enc = load_cllp_model(config["loc_enc_ckpt_path"], device=device)
        
    elif config.loc_embed_method == "space2vec":
        loc_enc = GlobalPositionEncDec(config)
        
    else:
        loc_enc = load_embed_model(config["loc_embed_method"], 
                                   config["loc_enc_ckpt_path"],
                                   exp_num=exp_num)
    
    # freeze the loc_enc
    if loc_enc is not None:
        for param in loc_enc.parameters():
            param.requires_grad = False

    if config.networkName == "mhsa":
        model = TransEncoder(config=config, total_loc_num=config.total_loc_num, loc_embbedder=loc_enc).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    # model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

    if logger is not None:
        logger.info(f"Total number of trainable parameters: {total_params}")
        logger.info(model)
        
    else:
        print(f"Total number of trainable parameters: {total_params}")
        print(model)

    return model


def get_dataloaders(config, logger, exp_num):
    # def get_dataloaders(config):

    dataset_train = sp_loc_dataset(
        config.source_root,
        data_type="train",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
        inductive_pct=config["inductive_pct"],
        get_testinductive=False,
        exp_num=exp_num
    )
    dataset_val = sp_loc_dataset(
        config.source_root,
        data_type="validation",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
        inductive_pct=config["inductive_pct"],
        get_testinductive=False,
        exp_num=exp_num
    )
    dataset_test = sp_loc_dataset(
        config.source_root,
        data_type="test",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
        inductive_pct=config["inductive_pct"],
        get_testinductive=False,
        exp_num=exp_num
    )

    kwds_train = {
        "shuffle": True,
        "num_workers": config["num_workers"],
        "drop_last": True,
        "batch_size": config["batch_size"],
        "pin_memory": False,
        # "sampler": train_sampler,
    }
    kwds_val = {
        # "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": False,
        # "sampler": val_sampler,
    }
    kwds_test = {
        # "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": False,
        # "sampler": test_sampler,
    }
    fn = collate_fn

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=fn, **kwds_test)

    if logger is not None:
        logger.info(f"The length of train_loader: {len(train_loader)}")
        logger.info(f"The length of val_loader: {len(val_loader)}") 
        logger.info(f"The length of test_loader: {len(test_loader)}") 
    else:
        print(f"The length of train_loader: {len(train_loader)}")
        print(f"The length of val_loader: {len(val_loader)}") 
        print(f"The length of test_loader: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def get_inductive_input_loader(config):
    dataset_testinductive = sp_loc_dataset(
        config.source_root,
        data_type="test",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
        inductive_pct=config["inductive_pct"],
        get_testinductive=True
    )
    kwds_testinductive = {
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": False,
    }
    fn = collate_fn
    testinductive_loader = torch.utils.data.DataLoader(dataset_testinductive, collate_fn=fn, **kwds_testinductive)
    print(f"The length of testinductive_loader: {len(testinductive_loader)}")
    return testinductive_loader


def get_logger(logger_name, log_dir, time_now):

    # Create a logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set its log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler and set its log level
    log_file = 'log_file' + time_now + '.log'
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def save_args(args, path):
    """
    Save args to json file
    """
    file_path = path + "/" + "args.txt"
    with open(file_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)


def get_hier_training_corpus(data_dir_root, dataset="fsq_nyc", inductive_pct=0, get_valid=False):
    """
    Get the training corpus from training data
    """
    if inductive_pct > 0:
        train_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_train_remove{inductive_pct}pct.pk")
        if get_valid:
            valid_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_validation_remove{inductive_pct}pct.pk")
    else:
        train_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_train.pk")
        if get_valid:
            valid_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_validation.pk")
    with open(train_file_path, "rb") as f:
        train_data = pickle.load(f)
    train_sentences = []
    train_users = []
    train_weekdays = []
    train_start_mins = []
    # train_corpus is a list of list of numbers
    for iter_dict in train_data:
        X = iter_dict["X"].tolist()
        Y = iter_dict["Y"]
        X.append(Y)
        train_sentences.append(X)
        train_users.append(iter_dict["user_X"])
        weekday_X = iter_dict["weekday_X"].tolist()
        weekday_Y = iter_dict["weekday_Y"]
        weekday_X.append(weekday_Y)
        train_weekdays.append(weekday_X)
        start_min_X = iter_dict["start_min_X"].tolist()
        start_min_Y = iter_dict["start_min_Y"]
        start_min_X.append(start_min_Y)
        train_start_mins.append(start_min_X)
    if get_valid:
        with open(valid_file_path, "rb") as f:
            valid_data = pickle.load(f)
        valid_sentences = []
        valid_users = []
        valid_weekdays = []
        valid_start_mins = []
        for iter_dict in valid_data:
            X = iter_dict["X"].tolist()
            Y = iter_dict["Y"]
            X.append(Y)
            valid_sentences.append(X)
            valid_users.append(iter_dict["user_X"])
            weekday_X = iter_dict["weekday_X"].tolist()
            weekday_Y = iter_dict["weekday_Y"]
            weekday_X.append(weekday_Y)
            valid_weekdays.append(weekday_X)
            start_min_X = iter_dict["start_min_X"].tolist()
            start_min_Y = iter_dict["start_min_Y"]
            start_min_X.append(start_min_Y)
            valid_start_mins.append(start_min_X)
        return train_sentences, train_users, train_weekdays, train_start_mins, valid_sentences, valid_users, valid_weekdays, valid_start_mins
    return train_sentences, train_users, train_weekdays, train_start_mins, None, None, None, None


def get_training_corpus(data_dir_root, dataset="fsq_nyc", inductive_pct=0, run_id=0):
    """
    Get the training corpus from training data
    """
    if inductive_pct > 0:
        train_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_train_remove{inductive_pct}pct_{run_id}.pk")
    else:
        train_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_train.pk")
    with open(train_file_path, "rb") as f:
        train_data = pickle.load(f)
    train_sentences = []
    train_users = []
    train_weekdays = []
    train_start_mins = []
    # train_corpus is a list of list of numbers
    for iter_dict in train_data:
        X = iter_dict["X"].tolist()
        Y = iter_dict["Y"]
        X.append(Y)
        train_sentences.append(X)
        train_users.append(iter_dict["user_X"])
        weekday_X = iter_dict["weekday_X"].tolist()
        weekday_Y = iter_dict["weekday_Y"]
        weekday_X.append(weekday_Y)
        train_weekdays.append(weekday_X)
        start_min_X = iter_dict["start_min_X"].tolist()
        start_min_Y = iter_dict["start_min_Y"]
        start_min_X.append(start_min_Y)
        train_start_mins.append(start_min_X)
    return train_sentences, train_users, train_weekdays, train_start_mins

def get_id2coor_df(data_dir_root, dataset="fsq_nyc"):
    loc_id_file = os.path.join(data_dir_root, dataset, f"{dataset}_locs.csv")
    id2coor_df = pd.read_csv(loc_id_file)
    # drop the location_id column and set the "location_id_num" as the index
    if "location_id" in id2coor_df.columns:
        id2coor_df.drop(columns=["location_id"], inplace=True)
    id2coor_df.set_index("location_id_num", inplace=True)
    # if the df contains columns "latitude" and "longitude", then we need to rename them
    if "latitude" in id2coor_df.columns:
        id2coor_df.rename(columns={"latitude": "lat", "longitude": "lng"}, inplace=True)
    return id2coor_df


def get_tale_training_corpus(data_dir_root, dataset="fsq_nyc", inductive_pct=0, run_id=0):
    if inductive_pct > 0:
        train_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_train_remove{inductive_pct}pct_{run_id}.pk")
    else:
        train_file_path = os.path.join(data_dir_root, dataset, f"{dataset}_7_train.pk")
    with open(train_file_path, "rb") as f:
        train_data = pickle.load(f)
    train_sentences = []
    train_timestamps = []
    # train_corpus is a list of list of numbers
    for iter_dict in train_data:
        X = iter_dict["X"].tolist()
        Y = iter_dict["Y"]
        X.append(Y)
        train_sentences.append(X)
        timestamp_X = iter_dict["timestamp_X"].tolist()
        timestamp_Y = iter_dict["timestamp_Y"]
        timestamp_X.append(timestamp_Y)
        train_timestamps.append(timestamp_X)
    return train_sentences, train_timestamps
