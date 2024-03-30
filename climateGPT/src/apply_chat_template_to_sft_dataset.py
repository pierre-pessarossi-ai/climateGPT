from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import datasets
import structlog
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizer

log = structlog.get_logger()

CHAT_TEMPLATE = """
<|im_start|>system
You are an expert on climate change and ecology.{eos_token}
<|im_end|>
<|im_start|>user
{instruction}{eos_token}<|im_end|>
<|im_start|>assistant
{answer}{eos_token}
"""

PATH_TO_SAVE = Path("climateGPT/data/templated_sft_dataset")


@dataclass
class TokenizerConfig:
    model_name: str
    model_max_length: int = 2048


@dataclass
class DatasetConfig:
    dataset_name: str = "pierre-pessarossi/climate-question-answers"


def load_tokenizer(model_name: str, model_max_length: int) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length > model_max_length:
        tokenizer.model_max_length = model_max_length

    return tokenizer


def load_sft_dataset(dataset_name: str) -> datasets.DatasetDict:
    return datasets.load_dataset(dataset_name)


def apply_chat_template_to_example(
    example: Dict[str, str], eos_token: str, template: str
) -> Dict[str, str]:
    example["eos_token"] = eos_token
    example["text"] = template.format(**example)
    return example


def apply_chat_template(
    raw_dataset: datasets.DatasetDict, tokenizer: PreTrainedTokenizer
):
    column_names = list(raw_dataset["train"].features)
    return raw_dataset.map(
        apply_chat_template_to_example,
        fn_kwargs={"template": CHAT_TEMPLATE, "eos_token": tokenizer.eos_token},
        remove_columns=column_names,
    )


def save_templated_dataset_locally(template_dataset: datasets.DatasetDict) -> None:
    template_dataset.save_to_disk(PATH_TO_SAVE)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Apply chat template to dataset and save locally"
    )
    parser.add_arguments(TokenizerConfig, dest="token_config")
    parser.add_arguments(DatasetConfig, dest="data_config")

    args = parser.parse_args()

    tokenizer = load_tokenizer(
        args.token_config.model_name, args.token_config.model_max_length
    )
    log.info(
        "Tokenizer loaded",
        model_name=args.token_config.model_name,
        max_length=args.token_config.model_max_length,
    )

    dataset = load_sft_dataset(args.data_config.dataset_name)
    log.info("Dataset loaded", dataset=dataset)

    dataset_w_template = apply_chat_template(dataset, tokenizer)
    log.info("Chat template applied to raw dataset")

    save_templated_dataset_locally(dataset_w_template)
    log.info("Templated dataset saved locally")
