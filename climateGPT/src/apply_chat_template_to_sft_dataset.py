from dataclasses import dataclass
from typing import Dict

import structlog
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer

from climateGPT.src.repositories import (
    ClimateSFTDatasetRepository,
    HubClimateSFTDatasetRepository,
    LocalTemplatedSFTDatasetRepository,
    TemplatedClimateSFTDatasetRepository,
)

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


@dataclass
class TokenizerConfig:
    model_name: str
    model_max_length: int = 2048


@dataclass
class DatasetConfig:
    climate_sft_dataset_repository: ClimateSFTDatasetRepository = (
        HubClimateSFTDatasetRepository
    )
    templated_climate_sft_dataset_repository: TemplatedClimateSFTDatasetRepository = (
        LocalTemplatedSFTDatasetRepository
    )


class ApplyChatTemplate:
    def __init__(
        self,
        climate_sft_dataset_repository: ClimateSFTDatasetRepository,
        templated_climate_sft_dataset_repository: TemplatedClimateSFTDatasetRepository,
        model_name: str,
        model_max_length: int = 2048,
    ) -> None:
        self.climate_sft_dataset_repository = climate_sft_dataset_repository
        self.templated_climate_sft_dataset_repository = (
            templated_climate_sft_dataset_repository
        )
        self.model_name = model_name
        self.model_max_length = model_max_length

    def load_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.tokenizer.model_max_length > self.model_max_length:
            self.tokenizer.model_max_length = self.model_max_length

        log.info(
            "Tokenizer loaded",
            model_name=args.token_config.model_name,
            max_length=args.token_config.model_max_length,
        )

    def load_sft_dataset(self) -> None:
        self.sft_dataset = self.climate_sft_dataset_repository().load()
        log.info("Dataset loaded", dataset=self.sft_dataset)

    @staticmethod
    def apply_chat_template_to_example(
        example: Dict[str, str], eos_token: str, template: str
    ) -> Dict[str, str]:
        example["eos_token"] = eos_token
        example["text"] = template.format(**example)
        return example

    def apply_chat_template(self) -> None:
        column_names = list(self.sft_dataset["train"].features)
        self.templated_sft_dataset = self.sft_dataset.map(
            ApplyChatTemplate.apply_chat_template_to_example,
            fn_kwargs={
                "template": CHAT_TEMPLATE,
                "eos_token": self.tokenizer.eos_token,
            },
            remove_columns=column_names,
        )
        log.info("Chat template applied to raw dataset")

    def save_templated_dataset(self) -> None:
        self.templated_climate_sft_dataset_repository().save(self.templated_sft_dataset)
        log.info("Templated dataset saved")

    def apply_chat_template_and_save_templated_dataset(self):
        self.load_tokenizer()
        self.load_sft_dataset()
        self.apply_chat_template()
        self.save_templated_dataset()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Apply chat template to dataset and save locally"
    )
    parser.add_arguments(TokenizerConfig, dest="token_config")
    parser.add_arguments(DatasetConfig, dest="data_config")

    args = parser.parse_args()

    apply_chat_template = ApplyChatTemplate(
        args.data_config.climate_sft_dataset_repository,
        args.data_config.templated_climate_sft_dataset_repository,
        args.token_config.model_name,
        args.token_config.model_max_length,
    )

    apply_chat_template.apply_chat_template_and_save_templated_dataset()
