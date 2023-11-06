"""Script to fine-tune a PLM for offensive language identification"""
import json
import argparse
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from shutil import rmtree

# since the evaluate module caused problems, using old datasets load_metric
from datasets import load_metric, DatasetDict
from util import parse_config, load_data, Data, OffensiveWordReplaceOption


accuracy = load_metric('accuracy')


def create_default_config(create_file: bool = False) -> dict[str, str]:
    """Creates a config file with default parameters, and returns them"""
    default_config = {
        'data_dir': './data',
        'preprocessed': True,  # True, False
        'replace_option': 'none',  # none, replace, remove
        'evaluation_set_while_training': 'dev',  # dev, test
        'model_id': 'bert-base-cased',  # model-id from huggingface.co/models
        'seed': None,  # int or None
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 5,
        'weight_decay': 0.01,
        'evaluation_strategy': 'epoch',  # epoch, steps
        'evaluation_set': 'dev' # dev or test
    }

    if create_file:
        with open('plm_config.json', 'x', encoding='utf-8') as config_file:
            json.dump(default_config, config_file, indent=4)

    return default_config


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments using the argparse parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config_file',
        default='plm_config.json',
        type=str,
        help='Config file to use (default plm_config.json)',
    )

    parser.add_argument(
        '-x',
        '--create_config',
        action='store_true',
        help='Creates a config file with default parameters (plm_config.json)',
    )

    parser.add_argument(
        '-o',
        '--output_folder',
        default='./model/',
        help='Where to output the model when fine-tuned (default ./model/)'
    )

    args = parser.parse_args()

    return args


def tokenize_data(data: dict, tokenizer: AutoTokenizer) -> str:
    """Tokenzes the data for use with the PLM"""

    return tokenizer(data['text'], padding=True, truncation=True)


def create_model(model_id: str):
    """Creates a model with the given parameters"""
    id2label = {0: 'NOT', 1: 'OFF'}
    label2id = {'NOT': 0, 'OFF': 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=id2label, label2id=label2id
    )

    return model


def compute_metrics(eval_pred):
    """Computes the accuracy of the model while training"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def finetune_model(
    dataset: DatasetDict,
    config: dict[str, str],
    tokenizer: AutoTokenizer,
    data_collator: DataCollatorWithPadding,
) -> Trainer:
    """Fine-tunes model with the parameters and data provided"""
    training_args = TrainingArguments(
        output_dir='./intermediate_models',
        seed=config['seed'] if config['seed'] else 42,
        # learning_rate=config['learning_rate'],
        # per_device_train_batch_size=16,
        # per_device_eval_batch_size=16,
        # num_train_epochs=config['epochs'],
        # weight_decay=config['weight_decay'],
        no_cuda=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model_init=lambda: create_model(config['model']),
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


def main():
    """Trains an lstm with the parameters provided in the config file"""
    args = parse_args()

    # if run with -x, create a config file with default parameters and exit
    if args.create_config:
        create_default_config(True)
        return

    config = parse_config(args.config_file, create_default_config())

    # load in data based on config options
    data = load_data(
        config['data_dir'],
        OffensiveWordReplaceOption.from_str(config['replace_option']),
        config['preprocessed'],
    )

    # convert Data object to DatasetDict
    dataset = data.to_dataset()

    # load tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # tokenize data
    dataset = dataset.map(
        tokenize_data, batched=True, fn_kwargs={'tokenizer': tokenizer}
    )

    trainer = finetune_model(dataset, config, tokenizer, data_collator)

    trainer.save_model(output_dir=args.output_folder)

    # remove all other intermediate models to save space
    rmtree('./intermediate_models')


if __name__ == '__main__':
    main()
