import json
import os

from tqdm import tqdm

from backdoor_inspection_new import *
import pandas as pd


def save_to_df(
    df,
    _anomaly_metric,
    dataset_name,
    num_classes,
    backdoor_settings,
    adaptive_attack_strategy=None,
):
    backdoor_type, trigger_type, source_class, target_class = backdoor_settings
    _raw_dict = {
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "backdoor_type": backdoor_type,
        "trigger_type": trigger_type,
        "source_class": source_class,
        "target_class": target_class,
        "anomaly_metric": _anomaly_metric,
    }
    if adaptive_attack_strategy is not None:
        _raw_dict["adaptive_attack_strategy"] = adaptive_attack_strategy
    df = df.append([_raw_dict], ignore_index=True)
    return df


def _inspect_one_model(
    saved_model_file, model_arch, opt, n_cls, size, method="FreeEagle"
):
    print(f"Inspecting model: {saved_model_file}")
    opt.inspect_layer_position = None
    opt.ckpt = saved_model_file
    opt.model = model_arch
    opt.n_cls = n_cls
    opt.size = size
    set_default_settings(opt)
    if method == "FreeEagle":
        _anomaly_metric = inspect_saved_model(opt)
    else:
        raise ValueError(f"Unimplemented method: {method}")
    return _anomaly_metric


method_name = "FreeEagle"
# root = 'D:/MyCodes/MLBackdoorDetection/saved_models'
root = "~/data/models"
# generate paths of saved model files
# datasets = ['imagenet_subset', 'gtsrb', 'cifar10', 'mnist']
datasets = ["cifar10"]

dataset_re_ag_dict = {"imagenet_subset": 10, "cifar10": 20, "gtsrb": 5, "mnist": 20}
dataset_re_sp_dict = {"imagenet_subset": 3, "cifar10": 8, "gtsrb": 4, "mnist": 8}

# dataset_arch_dict = {'imagenet_subset': 'resnet50', 'cifar10': 'vgg16', 'gtsrb': 'google_net', 'mnist': 'simple_cnn'}
dataset_arch_dict = {"cifar10": "resnet18"}
dataset_ncls_dict = {"imagenet_subset": 20, "cifar10": 10, "gtsrb": 43, "mnist": 10}
dataset_size_dict = {"imagenet_subset": 224, "cifar10": 32, "gtsrb": 32, "mnist": 28}
dataset_specific_backdoor_targeted_classes_dict = {
    "imagenet_subset": [0, 12, 14, 18],
    "cifar10": range(10),
    "gtsrb": [7, 8],
    "mnist": range(10),
}

trigger_types = ["patched_img", "blending_img", "filter_img"]


# generate empty df
df = pd.DataFrame(
    columns=[
        "dataset_name",
        "num_classes",
        "backdoor_type",
        "trigger_type",
        "source_class",
        "target_class",
        "anomaly_metric",
    ]
)
opt = parse_option()

model_num = 2
# check benign models
for dataset in datasets:
    model_arch = dataset_arch_dict[dataset]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]

    for benign_model_id in range(model_num):
        benign_model_id += 1
        saved_model_file = (
            f"{root}/clean_model/{dataset}/{model_arch}/{benign_model_id}.pth"
        )
        # try:
        _anomaly_metric = _inspect_one_model(
            saved_model_file, model_arch, opt, _n_cls, _size, method_name
        )
        # except FileNotFoundError:
        #     print(f'File not found.')
        #     continue
        # except RuntimeError:
        #     print('Ckpt file corrupted in multiple models 1.')
        #     continue
        backdoor_settings = ("None", "None", "None", "None")
        df = save_to_df(df, _anomaly_metric, dataset, _n_cls, backdoor_settings)
        filepath = f"results_benign_{method_name}.csv"
        df.to_csv(filepath, index=False)
        print(f"data frame saved to {filepath}.")


# generate empty df
df = pd.DataFrame(
    columns=[
        "dataset_name",
        "num_classes",
        "backdoor_type",
        "trigger_type",
        "source_class",
        "target_class",
        "anomaly_metric",
    ]
)
opt = parse_option()
attack_type_list = ["badnet"]  #

# check poisoned models
for dataset in datasets:
    REPEAT_ROUNDS_AGNOSTIC = dataset_re_ag_dict[dataset]
    REPEAT_ROUNDS_SPECIFIC = dataset_re_sp_dict[dataset]

    model_arch = dataset_arch_dict[dataset]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]

    poisoned_dataset = f"poisoned_{dataset}"
    time_cost_list = []
    for mal_model_id in range(model_num):
        start = time.time()
        for attack_type in attack_type_list:
            for trigger_type in trigger_types:
                # class agnostic backdoor
                for repeat_round_id in range(REPEAT_ROUNDS_AGNOSTIC):
                    for targeted_class in range(_n_cls):
                        saved_agnostic_poisoned_model_file = (
                            f"{root}/poison_model/"
                            f"image-{dataset}-{attack_type}-{model_arch}/"
                            f"{mal_model_id}.pth"
                        )
                        try:
                            _anomaly_metric = _inspect_one_model(
                                saved_agnostic_poisoned_model_file,
                                model_arch,
                                opt,
                                _n_cls,
                                _size,
                                method=method_name,
                            )
                        except FileNotFoundError:
                            print(f"File not found.")
                            break
                        except RuntimeError:
                            print("Ckpt file corrupted in multiple models 2.")
                            continue
                        _n_cls = dataset_ncls_dict[dataset]
                        backdoor_settings = (
                            "agnostic",
                            trigger_type,
                            "None",
                            targeted_class,
                        )
                        df = save_to_df(
                            df,
                            _anomaly_metric,
                            poisoned_dataset,
                            _n_cls,
                            backdoor_settings,
                        )
                        filepath = f"results_agnostic_{method_name}_{attack_type}.csv"
                        df.to_csv(filepath, index=False)
                        print(f"data frame saved to {filepath}.")
        end = time.time()
        poison_time_cost = end - start
        time_cost_list.apend(poison_time_cost)
    filepath = f"results_time_cost_{dataset}.json"
    with open(filepath, "w") as f:
        json.dump(f, filepath)
