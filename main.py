import os
import argparse
import json

from datetime import datetime

import pandas as pd
import torch

from easydict import EasyDict as edict

from utils.utils import load_config, setup_seed, get_trained_nets, \
get_test_result, get_dataloaders, get_models, get_logger, get_inductive_input_loader

setup_seed(42)


def init_save_path(config, time_now, i):
    """define the path to save, and save the configuration file."""
    networkName = f"{config.dataset}_{config.loc_embed_method}_{config.networkName}"
    if config["inductive_pct"] > 0:
        log_dir = os.path.join(config["inductive_save_root"], f"{networkName}_{time_now}_{str(i)}")
    else:
        log_dir = os.path.join(config.save_root, f"{networkName}_{time_now}_{str(i)}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = get_logger('my_logger', log_dir, time_now)

    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    if config["inductive_pct"] > 0:
        logger.info(f"Begin the {i}th run of the inductive experiment.")
    else:
        logger.info(f"Begin the {i}th run of the experiment.")
    return logger, log_dir


def single_run(train_loader, val_loader, test_loader, 
               config, device, log_dir, logger, exp_num=0):
    result_ls = []

    # get modelp
    if config['inductive_pct'] > 0:
        model = get_models(config, device, logger, exp_num=exp_num)
    else:
        model = get_models(config, device, logger, exp_num=0)

    # train
    model, perf = get_trained_nets(config, model, train_loader, val_loader, test_loader, device, log_dir, logger)
    result_ls.append(perf)

    # test
    perf, individual_test_perf, df_res = get_test_result(config, model, test_loader, device, logger)
    individual_test_perf.to_csv(os.path.join(log_dir, "user_detail.csv"))
    df_res.to_csv(os.path.join(log_dir, "df_res.csv"), index=False)

    result_ls.append(perf)

    return result_ls, model


def main(config, time_now):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    result_ls = []
    result_ls_inductive = []
    for i in range(5):
        # save the conf
        logger, log_dir = init_save_path(config, time_now, i)

        train_loader, val_loader, test_loader = get_dataloaders(config=config, logger=logger, exp_num=i)

        # Get the inductive test input
        # if config["inductive_pct"] > 0:
        #     test_inductive_loader = get_inductive_input_loader(config)

        res_single, best_model = single_run(train_loader, val_loader, test_loader, config, device, log_dir, logger, exp_num=i)

        if config["inductive_pct"] > 0:
            # test zeroshot
            perf, individual_test_perf, df_res = get_test_result(config, best_model, test_loader, device, logger)
            individual_test_perf.to_csv(os.path.join(log_dir, "user_detail_inductive.csv"))
            df_res.to_csv(os.path.join(log_dir, "df_res_inductive.csv"), index=False)

            result_ls_inductive.append(perf)
            logger.info(f"Inductive performance: {perf}")

        result_ls.extend(res_single)

        # Close and remove all handlers associated with the logger
        for handler in logger.handlers[:]:  # Make a copy of the list
            handler.close()
            logger.removeHandler(handler)

    result_df = pd.DataFrame(result_ls)
    # train_type = "default"
    filename = os.path.join(
        log_dir,
        f"{config.dataset}_{config.networkName}_{time_now}.csv",
    )
    result_df.to_csv(filename, index=False)

    if config["inductive_pct"] > 0:
        result_df_inductive = pd.DataFrame(result_ls_inductive)
        filename = os.path.join(
            log_dir,
            f"{config.dataset}_{config.networkName}_{time_now}_inductive.csv",
        )
        result_df_inductive.to_csv(filename, index=False)
    
    print("Done!")


if __name__ == "__main__":

    # time_now = int(datetime.now().timestamp())
    # Init time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    # Load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, nargs="?", help=" Config file path.", default="config/foursquare/transformer.yml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    # path = os.path.join(config.save_root,
    #                     formatted_datetime)  # unique checkpoint saving path
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # logger = get_logger('my_logger', log_dir=path)
    # save_args(args=config, path=path)

    main(config, time_now=formatted_datetime)
