# import os
import time
import torch
import sklearn  # Although the package is not used, it needs to be imported, otherwise, in Linux environments,
# there are sometimes errors in the allocation of the library, for reasons that remain to be seen.
import transformers
# from transformers import AutoModel, AutoConfig
from dataclasses import dataclass, field

from dataloader.dataloader import infidelity_site_loader_csv, DataCollatorForInfidelityDataset
from utils.utils import set_global_random_seed, setup_path, save_test_results_and_best_model, check_args
from models.TDSitePre import TDSitePre
from models.evaluation_method.eval_infidelity import CalculateMetric


def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return hours, minutes, seconds


def run(base_args, extension_args):
    check_args(extension_args)

    # Check available equipment
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cnt = torch.cuda.device_count()
    print("Total {} GPUs available to use!".format(device_cnt))

    # Parameter consistency check and adjustment
    if extension_args.predicting_content == 'sequence' or extension_args.predicting_content == 'seq_token':
        base_args.metric_for_best_model = 'seq_f1'
        base_args.greater_is_better = True
    else:  # predicting_content: token
        base_args.metric_for_best_model = 'tok_f1'
        base_args.greater_is_better = True

    assert extension_args.work_mode in ['training', 'prediction']

    # Loading the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(extension_args.model_name_or_path,
                                              use_fast=True,
                                              trust_remote_code=True)
    # Loading the model
    model = TDSitePre(extra_config=extension_args)

    if extension_args.work_mode == 'training':
        # Update/configure the saved path based on the training parameters
        base_args.output_dir, base_args.run_name = setup_path(base_args, extension_args)
        # Set the global random seed
        set_global_random_seed(base_args.seed)

        # Constructing the dataset
        train_dataset = infidelity_site_loader_csv(extension_args, load_type='train', tokenizer=tokenizer)
        val_dataset = infidelity_site_loader_csv(extension_args, load_type='val', tokenizer=tokenizer)
        test_dataset = infidelity_site_loader_csv(extension_args, load_type='test', tokenizer=tokenizer)

        # Define the data collector
        data_collator = DataCollatorForInfidelityDataset(tokenizer=tokenizer)

        # define trainer
        trainer = transformers.Trainer(model=model,
                          tokenizer=tokenizer,
                          args=base_args,
                          compute_metrics=CalculateMetric(config=extension_args),
                          train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          data_collator=data_collator)

        # Start training and validation
        trainer.train()

        # Test the model and save the test results and the optimal model related files
        save_test_results_and_best_model(trainer=trainer, base_args=base_args, extension_args=extension_args,
                                         test_dataset=test_dataset)
    else:  # TODO prediction only
        # Since the interpretation method does not use the region identifier model for the time being,
        # the prediction here is not yet realized.
        print('The part of the region identifier model that is only used for prediction has not been implemented')

    return None


@dataclass
class TrainingArgsBase(transformers.TrainingArguments):
    # Basic parameter setting
    seed: int = field(default=1)
    learning_rate: float = field(default=1e-4)  # Initial learning rate setting for the pretrained structure
    num_train_epochs: int = field(default=1)

    optim: str = field(default="adamw_torch")  # Optimizer selection
    gradient_accumulation_steps: int = field(default=1)
    fp16: bool = field(default=False)

    # Output Settings
    output_dir: str = field(default=r"F:/output/tdi_results")
    run_name: str = field(default="run")  # The name of the folder to output the file
    logging_steps: int = field(default=10)
    log_level: str = field(default="debug")  # Control log output ['info', 'debug']

    # Batch size Settings
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=128)

    save_steps: int = field(default=2)
    save_total_limit: int = field(default=2)
    save_strategy: str = field(default='steps')

    eval_steps: int = field(default=2)
    evaluation_strategy: str = field(default="steps")

    warmup_steps: int = field(default=2)
    weight_decay: float = field(default=0.01)

    load_best_model_at_end: bool = field(default=True)

    dataloader_pin_memory: bool = field(default=True)
    overwrite_output_dir: bool = field(default=True)

    # Evaluation metrics for the optimal model
    metric_for_best_model: str = field(default="loss")  # seq_f1
    greater_is_better: bool = field(default=False)  # True means that a larger indicator is better, False the opposite.


@dataclass
class TrainingArgsExtension:
    # lr_scale: int = field(default=100)

    # 补充设置
    model_name_or_path: str = field(default=r"...\DNABERT-2-117M")  # DNABERT2 model weight path
    max_length: int = field(default=60, metadata={"help": "Maximum number of tokens."})
    # cache_dir: Optional[str] = field(default=None)

    work_mode: str = field(default='training')  # training or prediction
    classifier: str = field(default='bert_token')  # ['bert_token']
    loss_function: str = field(default='WBCE')

    data_path: str = field(default=r"F:\output\data\10000_s1n200l1_ed_mdp_df", metadata={"help": "Path to the data."})
    train_data_name: str = field(default='train.csv')
    val_data_name: str = field(default='dev.csv')
    test_data_name: str = field(default='test.csv')

    model_architecture: str = field(default='traditional_learning')
    base_frame: str = field(default='bert')  # ['bert', 'bilstm']

    # ['batch_sequence_weighted', 'single_sequence_weighted', 'ssw_only_consider_pos']
    WBCE_type: str = field(default='ssw_only_consider_pos')
    bce_params_use_dynamic_pos_weight: bool = field(default=True)
    bce_params_pos_weight: float = field(default=1.0)

    # Specific parameters needed for different losses
    # Dice loss requires parameters
    dl_params_smooth: float = field(default=1.0)
    dl_params_ohem_ratio: float = field(default=0.5)
    dl_params_alpha: float = field(default=0.2)
    dl_params_square_denominator: bool = field(default=True)

    # Configure multi-token prediction related parameters
    num_merge_token: int = field(default=2)  # The default value is 1, in which case it is a single-token prediction
    merge_token_gap: int = field(default=1)  # Sliding spacing of two adjacent m-tokens
    keep_tail_tokens: bool = field(default=True)
    merge_method: str = field(default='mean_max')  # ['mean', 'max', 'mean_max', 'bilstm']  TODO: max, self_attention
    post_mm_method: str = field(default='none')  # ['none', 'bilstm']

    # Prediction content: Indicates which level of infidelity prediction needs to be included
    # sequence denotes only predicting the infidelity at the sequence level
    # token denotes the infidelity that only predicts the token level
    # seq_token indicates the infidelity at the prediction sequence and token level
    predicting_content: str = field(default='sequence')  # ['sequence', 'token', 'seq_token']

    seq_loss_coefficient: float = field(default=0.5)  # 0.0-1.0 indicates that the proportion of sequence loss in the total loss is only meaningful if the predicting_content is seq_token.

    classifier_order: str = field(default='parallel')  # ['serial', 'parallel'] Denote the sequential relationship between seq and tok classifiers

    # eval_seq_tok_correlation makes sense when classifier_order is parallel
    eval_seq_tok_correlation: bool = field(default=True)  # True means that only samples with infidelity in the seq prediction are considered when evaluating tok



if __name__ == '__main__':
    start_time = time.time()

    parser = transformers.HfArgumentParser((TrainingArgsBase, TrainingArgsExtension))
    ta_base, ta_extension = parser.parse_args_into_dataclasses()

    run(ta_base, ta_extension)

    end_time = time.time()  # Record the end time of the program
    run_time = end_time - start_time  # Calculate the running time in seconds
    print("program start time:", time.ctime(start_time))
    print("program end time:", time.ctime(end_time))
    hours, minutes, seconds = convert_seconds(run_time)
    print(f"Run time: {hours}hours {minutes}minutes {seconds}seconds")







