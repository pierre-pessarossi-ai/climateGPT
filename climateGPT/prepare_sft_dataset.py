import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog
from datasets import load_dataset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from simple_parsing import ArgumentParser
from tqdm import tqdm

log = structlog.get_logger()

LOCAL_CACHE = "climateGPT/data/sft_dataset"

DATASET_HF_ADDRESS = "pierre-pessarossi/wikipedia-climate-data"

MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "open-mixtral-8x7b",
]

OPENAI_MODELS = ["gpt-3.5-turbo-0125"]


@dataclass
class PrepareDataConfig:
    prompt_template_name: str = "sft_dataset.txt"
    model_name: str = "mistral-large-latest"
    dataset_hf_address: str = DATASET_HF_ADDRESS
    local_cache: str = LOCAL_CACHE
    max_range: int = -1


def create_key_from_title(title: str) -> str:
    hash_obj = hashlib.sha256(title.encode())
    return hash_obj.hexdigest()[:16]


def call_mistral(prompt, model: str = "mistral-large-latest") -> str:
    api_key = os.environ["MISTRAL_API_KEY"]
    if api_key is None:
        raise ValueError("Missing Mistral API key in env. variables.")
    client = MistralClient(api_key=api_key)
    messages = [ChatMessage(role="user", content=prompt)]
    chat_response = client.chat(
        model=model,
        messages=messages,
    )
    return chat_response.choices[0].message.content


def call_openai(prompt, model: str = "gpt-3.5-turbo-0125") -> str:
    ...


class PrepareSFTDataset:
    def __init__(
        self,
        dataset_hf_address: str,
        model_name: str,
        prompt_template_name: str,
        local_cache: Path,
    ) -> None:
        self.dataset_hf_address = dataset_hf_address
        self.model_name = model_name
        self.prompt_template_name = prompt_template_name
        self.load_prompt_template()
        self.local_cache = Path(local_cache) / self.model_name.replace("-", "_")
        self.local_cache.mkdir(exist_ok=True, parents=True)
        self.load_dataset_from_hf()

    def load_dataset_from_hf(self) -> None:
        dataset = load_dataset(self.dataset_hf_address)
        dataset = dataset["train"].to_pandas()
        dataset = dataset.assign(
            content_len=lambda x: x["content"].apply(len),
            content_len_group=lambda x: pd.qcut(
                x["content_len"], 10, labels=range(1, 11)
            ),
        ).reset_index(drop=True)
        self.dataset = dataset

    def load_prompt_template(self) -> None:
        with open(f"climateGPT/prompt_templates/{self.prompt_template_name}", "r") as f:
            self.prompt_template = f.read()

    def format_prompt(self, **kwargs) -> str:
        return self.prompt_template.format(**kwargs)

    def create_instruction_answer(self, index: int) -> None:
        cached = [
            filename.name.replace(".json", "")
            for filename in self.local_cache.glob("*.json")
        ]
        row_of_interest = self.dataset.loc[index]
        prompt = self.format_prompt(
            **{
                "title": row_of_interest["title"],
                "content": row_of_interest["content"],
                "number_of_questions": row_of_interest["content_len_group"],
            },
        )
        encoded_title = create_key_from_title(row_of_interest["title"])

        if encoded_title in cached:
            log.info("Already in cache")
        else:
            log.info("Prompt formatted")
            if self.model_name in MISTRAL_MODELS:
                answer_i = call_mistral(prompt, self.model_name)
            elif self.model_name == "gpt-4":
                raise ValueError("Call to GPT4 not yet implemented")
            log.info("Answer received")
            with open(
                self.local_cache / f"""{encoded_title}.json""",
                "w",
            ) as f:
                for line in answer_i:
                    f.write(line)
            log.info("Entry cached")

    def __call__(self, max_range: int) -> None:
        if max_range == -1 or max_range > len(self.dataset):
            max_range = len(self.dataset)

        for i in tqdm(range(max_range)):
            self.create_instruction_answer(i)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a dataset for SFT of an LLM.")
    parser.add_arguments(PrepareDataConfig, dest="data_config")

    args = parser.parse_args()

    prepare_sft_dataset = PrepareSFTDataset(
        dataset_hf_address=args.data_config.dataset_hf_address,
        model_name=args.data_config.model_name,
        prompt_template_name=args.data_config.prompt_template_name,
        local_cache=args.data_config.local_cache,
    )

    prepare_sft_dataset(max_range=args.data_config.max_range)
