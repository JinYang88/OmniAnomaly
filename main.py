# -*- coding: utf-8 -*-
import logging
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from pprint import pformat, pprint

import numpy as np
import pandas as pd
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import (
    get_variables_as_dict,
    register_config_arguments,
    Config,
)

from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.training import Trainer
from omni_anomaly.utils import (
    get_data_dim,
    get_data,
    save_z,
    iter_thresholds,
    load_dataset,
    subdatasets,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from IPython import embed


class ExpConfig(Config):
    # dataset configuration
    dataset = "SMAP"
    # dataset = "SMAP"
    x_dim = get_data_dim(dataset)

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = "GRU"  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = "nf"  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 1
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 64
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -400.0
    bf_search_max = 400.0
    bf_search_step_size = 1.0

    valid_step_freq = 100
    gradient_clip_norm = 10.0

    early_stop = False  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.07

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = "model"
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = "result"  # Where to save the result file
    train_score_filename = "train_score.pkl"
    test_score_filename = "test_score.pkl"


def main(dataset, subdataset):
    logging.basicConfig(
        level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # # prepare the data
    # (x_train, _), (x_test, y_test) = get_data(
    #     config.dataset,
    #     config.max_train_size,
    #     config.max_test_size,
    #     train_start=config.train_start,
    #     test_start=config.test_start,
    # )

    (x_train, _), (x_test, y_test) = load_dataset(dataset, subdataset)

    tf.reset_default_graph()
    # construct the model under `variable_scope` named 'model'
    with tf.variable_scope("model") as model_vs:
        model = OmniAnomaly(config=config, name="model")

        # construct the trainer
        trainer = Trainer(
            model=model,
            model_vs=model_vs,
            max_epoch=config.max_epoch,
            batch_size=config.batch_size,
            valid_batch_size=config.test_batch_size,
            initial_lr=config.initial_lr,
            lr_anneal_epochs=config.lr_anneal_epoch_freq,
            lr_anneal_factor=config.lr_anneal_factor,
            grad_clip_norm=config.gradient_clip_norm,
            valid_step_freq=config.valid_step_freq,
        )

        # construct the predictor
        predictor = Predictor(
            model,
            batch_size=config.batch_size,
            n_z=config.test_n_z,
            last_point_only=True,
        )

        with tf.Session().as_default():

            if config.restore_dir is not None:
                # Restore variables from `save_dir`.
                saver = VariableSaver(
                    get_variables_as_dict(model_vs), config.restore_dir
                )
                saver.restore()

            if config.max_epoch > 0:
                # train the model
                train_start = time.time()
                best_valid_metrics = trainer.fit(x_train)
                train_time = time.time() - train_start
                # best_valid_metrics.update({"train_time": train_time})
            else:
                best_valid_metrics = {}

            # get score of train set for POT algorithm
            train_score, train_z, train_pred_speed = predictor.get_score(x_train)
            if config.train_score_filename is not None:
                with open(
                    os.path.join(config.result_dir, config.train_score_filename), "wb"
                ) as file:
                    pickle.dump(train_score, file)
            if config.save_z:
                save_z(train_z, "train_z")

            if x_test is not None:
                # get score of test set
                test_start = time.time()
                test_score, test_z, pred_speed = predictor.get_score(x_test)
                test_time = time.time() - test_start
                if config.save_z:
                    save_z(test_z, "test_z")
                best_valid_metrics.update(
                    {"pred_time": pred_speed, "pred_total_time": test_time}
                )
                if config.test_score_filename is not None:
                    with open(
                        os.path.join(config.result_dir, config.test_score_filename),
                        "wb",
                    ) as file:
                        pickle.dump(test_score, file)

                if y_test is not None and len(y_test) >= len(test_score):
                    if config.get_score_on_dim:
                        # get the joint score
                        test_score = np.sum(test_score, axis=-1)
                        train_score = np.sum(train_score, axis=-1)

                    # get best f1
                    t, th = bf_search(
                        test_score,
                        y_test[-len(test_score) :],
                        start=config.bf_search_min,
                        end=config.bf_search_max,
                        step_num=int(
                            abs(config.bf_search_max - config.bf_search_min)
                            / config.bf_search_step_size
                        ),
                        display_freq=50,
                    )
                    # get pot results
                    pot_result = pot_eval(
                        train_score,
                        test_score,
                        y_test[-len(test_score) :],
                        level=config.level,
                    )

                    # output the results
                    best_valid_metrics.update(
                        {
                            "best-f1": t[0],
                            "precision": t[1],
                            "recall": t[2],
                            "TP": t[3],
                            "TN": t[4],
                            "FP": t[5],
                            "FN": t[6],
                            "latency": t[-1],
                            "threshold": th,
                            "test_score": test_score,
                            "labels": y_test[-len(test_score) :],
                        }
                    )
                    best_valid_metrics.update(pot_result)
                results.update_metrics(best_valid_metrics)

            if config.save_dir is not None:
                # save the variables
                var_dict = get_variables_as_dict(model_vs)
                saver = VariableSaver(var_dict, config.save_dir)
                saver.save()
            print("=" * 30 + "result" + "=" * 30)
            pprint(best_valid_metrics)

            return best_valid_metrics


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-ws", "--window_size", default=32, required=False)

    args = arg_parser.parse_args(sys.argv[1:])

    dataset = "SMD"

    detail_dir = "./details"
    os.makedirs(detail_dir, exist_ok=True)

    start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    config = ExpConfig()
    # parse the arguments
    register_config_arguments(config, arg_parser)
    for subdataset in subdatasets[dataset][0:2]:
        # get config obj
        config.window_length = args.window_size
        config.x_dim = get_data_dim(dataset)

        # print_with_title("Configurations", pformat(config.to_dict()), after="\n")
        # open the result object and prepare for result directories if specified
        results = MLResults(config.result_dir)
        results.save_config(config)  # save experiment settings for review
        results.make_dirs(config.save_dir, exist_ok=True)

        records = []
        with warnings.catch_warnings():
            # suppress DeprecationWarning from NumPy caused by codes in TensorFlow-Probability
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module="numpy"
            )

            best_valid_metrics = main(dataset, subdataset)

            with open("running_time.txt", "a+") as fw:
                tr_time = best_valid_metrics["train_time"]
                te_time = best_valid_metrics["pred_time"]
                fw.write("window_size: {} ".format(config.window_length))
                fw.write("train: {:.4f} ".format(tr_time))
                fw.write("test: {:.4f}\n".format(te_time))

            score = best_valid_metrics["test_score"]
            anomaly_label = best_valid_metrics["labels"]
            best_f1, best_theta, pred_adjusted, best_raw = iter_thresholds(
                score, anomaly_label
            )
            raw_f1 = f1_score(best_raw, anomaly_label)
            ps_adjusted = precision_score(pred_adjusted, anomaly_label)
            rc_adjusted = recall_score(pred_adjusted, anomaly_label)
            auc = roc_auc_score(anomaly_label, score)
            record = {
                "score": score,
                "pred_raw": best_raw,
                "pred": pred_adjusted,
                "anomaly_label": anomaly_label,
                "theta": best_theta,
                "AUC": auc,
                "F1": raw_f1,
                "F1_adj": best_f1,
                "PS_adj": ps_adjusted,
                "RC_adj": rc_adjusted,
                "train_time": best_valid_metrics["train_time"],
                "test_time": best_valid_metrics["pred_time"],
            }
            records.append(record)
    records = pd.DataFrame(records)
    records.to_csv(
        "./{}/{}-{}-all.csv".format(detail_dir, dataset, start_time),
        index=False,
    )

    log = "{}\t{}\t{}\t{}\tAUC-{:.4f}\tF1-{:.4f}\tF1_adj-{:.4f}\tPS_adj-{:.4f}\tRC_adj-{:.4f}\ttrain-{}s\ttest-{}s\n".format(
        start_time,
        args.window_size,
        "Omni",
        dataset + "_all",
        records["AUC"].mean(),
        records["F1"].mean(),
        records["F1_adj"].mean(),
        records["PS_adj"].mean(),
        records["RC_adj"].mean(),
        records["train_time"].sum(),
        records["test_time"].sum(),
    )
    with open("./total_results.csv", "a+") as fw:
        fw.write(log)