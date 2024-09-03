import csv

import datasets
import portalocker
import json
import os
import random
import torch
import numpy as np
import transformers
from dataclasses import asdict


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Note: usually benchmark should be False for reproducibility
    transformers.set_seed(seed)

    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup_path(base_args, extension_args):
    resPath = f'epoch-{base_args.num_train_epochs}'
    resPath += f'.lr-{base_args.learning_rate}'
    resPath += f'.wus-{base_args.warmup_steps}'
    # resPath += f'.lr_scale-{extension_args.lr_scale}'
    resPath += f'.bs-{base_args.train_batch_size}'
    # resPath += f'.ml_{args.max_length}'  # max length
    # resPath += f'.tmp{args.temperature}'
    resPath += f'.seed-{base_args.seed}'
    resPath += f'.cf-{extension_args.classifier}'  # classifier
    resPath += f'.co-{extension_args.classifier_order}'  # classifier order
    resPath += f'.lf-{extension_args.loss_function}'  # loss_function

    # Add the WBCE type parameter flag
    # [batch_sequence_weighted, single_sequence_weighted, ssw_only_consider_pos]
    if extension_args.WBCE_type == 'batch_sequence_weighted':
        resPath += f'.wbce-bsw'
    elif extension_args.WBCE_type == 'single_sequence_weighted':
        resPath += f'.wbce-ssw'
    elif extension_args.WBCE_type == 'ssw_only_consider_pos':
        resPath += f'.wbce-ssw_ocp'
    else:
        raise ValueError('WBCE_type should in [batch_sequence_weighted, '
                         'single_sequence_weighted, ssw_only_consider_pos]')

    if extension_args.model_architecture == 'traditional_learning':
        resPath += f'.ma-tl'
    else:
        resPath += f'.ma-ol'  # other learning

    # Add the base_frame parameter flag
    resPath += f'.bf-{extension_args.base_frame}'

    # Add the num_merge_token parameter flag
    resPath += f'.nmt-{extension_args.num_merge_token}'
    # Add the keep_tail_tokens parameter flag
    if extension_args.keep_tail_tokens:
        resPath += f'.ktt'
    # Add the merge_method parameter flag
    resPath += f'.mm-{extension_args.merge_method}'
    # Add the post_mm_method parameter flag
    resPath += f'.pmm-{extension_args.post_mm_method}'

    # Add the merge_token_gap parameter flag
    if extension_args.merge_token_gap <= 0 or extension_args.merge_token_gap == extension_args.num_merge_token:
        resPath += f'.mtg-nmt'
    else:
        resPath += f'.mtg-{extension_args.merge_token_gap}'

    # Add the predicting_content parameter flag
    if extension_args.predicting_content == 'sequence':
        pc = 'seq'
    elif extension_args.predicting_content == 'token':
        pc = 'tok'
    else:
        pc = 'seq_tok'
    resPath += f'.pc-{pc}'

    # Add the seq_loss_coefficient parameter flag
    resPath += '.slc-{}'.format(extension_args.seq_loss_coefficient)

    if extension_args.eval_seq_tok_correlation:
        resPath += f'.eval-stc'

    # Get the last folder name of the path
    last, second_last = get_second_last_folder_name(extension_args.data_path)

    # Contains sequence length and extraction interval
    signal1 = second_last.split('_')[-2] if len(second_last.split('_')) >= 2 else ''
    signal2 = last.split('_')[1]
    signal3 = last.split('_')[-1]

    dataset_name = '_'.join([signal1, signal2, signal3])

    # Save some information
    base_args.initial_output_dir = base_args.output_dir
    base_args.signal_dataset_name = dataset_name
    base_args.signal_resPath = resPath

    output_dir = os.path.join(base_args.output_dir, dataset_name, resPath)

    print(f'output_dir path (corresponding dataset): {output_dir}')
    print(f'run_name filename (corresponding training configuration): {resPath}')

    return output_dir, resPath


def get_second_last_folder_name(path):
    # Normalize the path first
    norm_path = os.path.normpath(path)
    # Get the last part of the path and the rest
    path, last = os.path.split(norm_path)
    # Get the last part of the path again, i.e. the penultimate part
    path, second_last = os.path.split(path)
    return last, second_last


def check_args(extension_args):
    assert extension_args.work_mode in ['training']
    assert extension_args.classifier in ['bert_token']
    assert extension_args.loss_function in ['WBCE']
    assert extension_args.model_architecture in ['traditional_learning']
    assert extension_args.base_frame in ['bert', 'bilstm']
    assert extension_args.WBCE_type in ['batch_sequence_weighted', 'single_sequence_weighted', 'ssw_only_consider_pos']
    assert extension_args.predicting_content in ['sequence', 'token', 'seq_token']
    assert 0.0 <= extension_args.seq_loss_coefficient <= 1.0
    assert extension_args.classifier_order in ['serial', 'parallel']
    assert extension_args.merge_method in ['mean', 'max', 'mean_max', 'bilstm']
    assert extension_args.post_mm_method in ['none', 'bilstm']

    assert extension_args.num_merge_token >= 1, 'The num_merge_token parameter must be greater than or equal to 1.'

    # extension_args.merge_token_gap: Recommended value 1 or extension_args.num_merge_token
    if extension_args.merge_token_gap <= 0:
        extension_args.merge_token_gap = extension_args.num_merge_token


def lock_file(file):
    portalocker.lock(file, portalocker.LOCK_EX)


def unlock_file(file):
    portalocker.unlock(file)


def save_test_results_and_best_model(trainer: transformers.Trainer, base_args, extension_args, test_dataset):
    # Save the training state to the path: output_dir + trainer_state.json
    trainer.save_state()

    # Loading the trained optimal model
    # Getting the optimal model path
    best_model_path = trainer.state.best_model_checkpoint
    print('best_model_path: {}'.format(best_model_path))

    # Load the best model, don't use it for now, because by default trainer.train() will load the best model (load_best_model_at_end=True).
    # best_model = trainer.model_class.from_pretrained(best_model_path)

    # Generalisation ability test
    # Evaluation using optimal models
    results = trainer.evaluate(eval_dataset=test_dataset)
    results_path = base_args.output_dir

    # Save test results to the output directory
    with open(os.path.join(results_path, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Output to terminal
    print("The final model test results are shown below:")
    for result in results:
        print(f"{result}:{results[result]}")

    # Store test results as an append to a specified csv file (optional)
    # ----------------------------------------------------------------
    # List of keys to be extracted in order
    keys_seq = ['precision', 'recall', 'f1', 'auprc', 'matthews_correlation', 'loss', 'accuracy', 'auroc']
    keys_tok = ['precision', 'recall', 'f1', 'auprc', 'matthews_correlation', 'DLR', 'accuracy', 'auroc']
    keys = ['train_flag', 'dataset_flag'] + ['eval_seq_' + k if k != 'loss' else 'eval_loss' for k in keys_seq] + ['eval_tok_' + k for k in keys_tok]

    # The results of running tests are saved in one place, and at the same time,
    # they are organized in a specified key order, which facilitates statistical analysis of the results.
    test_results = os.path.join(base_args.initial_output_dir, 'test_results.csv')
    results['dataset_flag'] = base_args.signal_dataset_name
    results['train_flag'] = base_args.signal_resPath

    # Open CSV files in append mode
    with open(test_results, mode='a', newline='') as file:
        # Obtaining a file lock
        lock_file(file)

        writer = csv.DictWriter(file, fieldnames=keys)

        # Check if the file is empty
        file_empty = file.tell() == 0
        if file_empty:
            # Write to header line (optional)
            writer.writeheader()

        # Extracts the values in the result dictionary in the order of keys and appends them to a new line.
        row = {}
        for key in keys:
            if key in results:
                if isinstance(results[key], float):
                    if key == 'eval_tok_DLR':
                        row[key] = f'{results[key]:.1f}'
                    else:
                        row[key] = f'{results[key]:.4f}'
                else:
                    row[key] = results[key]
            else:
                row[key] = None

        # row = {key: f'{results[key]:.4f}' if key in results else None for key in keys}  # here, only consider float
        writer.writerow(row)

        # Releasing a file lock
        unlock_file(file)
    # ----------------------------------------------------------------

    # Save the best model
    best_mp = os.path.join(base_args.output_dir, 'best_model')
    trainer.save_model(best_mp)

    # Saving the BERT model configuration to the optimal model path
    bert_config = transformers.AutoConfig.from_pretrained(extension_args.model_name_or_path)
    bert_config.save_pretrained(best_mp)

    # Save the configuration of this training to the output directory
    with open(os.path.join(results_path, "base_args.json"), "w") as f:
        json.dump(asdict(base_args), f, indent=4)
    with open(os.path.join(results_path, "extension_args.json"), "w") as f:
        json.dump(asdict(extension_args), f, indent=4)

