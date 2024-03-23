import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path

import datasets
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


@dataclass
class PushToHuggingFaceHubConfig:
    local_folder: str
    user_name_on_huggingface: str
    dataset_name_on_huggingface: str
    random_seed: int = 498
    test_split: float = 0.2


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


class PushDatasetToHuggingFaceHub:
    def __init__(
        self,
        local_folder: str,
        user_name_on_huggingface: str,
        dataset_name_on_huggingface: str,
        random_seed: int = 498,
        test_split: float = 0.2,
    ) -> None:
        self.local_folder = Path(local_folder)
        self.files = [file.name for file in self.local_folder.glob("*.json")]
        self.user_name_on_huggingface = user_name_on_huggingface
        self.dataset_name_on_huggingface = dataset_name_on_huggingface
        self.random_seed = random_seed
        self.test_split = test_split

    def load_data_in_pandas_df(self) -> None:
        self.df = pd.concat(
            [pd.read_json(self.local_folder / file, lines=True) for file in self.files],
            axis=0,
            ignore_index=True,
        )
        log.info(f"Pandas dataframe loaded of shape: {self.df.shape}")

    def perform_train_test_split(self) -> None:
        full_index = self.df.index.to_list()
        random.seed(self.random_seed)
        random.shuffle(full_index)
        test_size = int(self.test_split * len(self.df))
        self.test_index = full_index[:test_size]
        self.train_index = full_index[test_size:]
        test_share = len(self.test_index) / (
            len(self.train_index) + len(self.test_index)
        )
        log.info(f"""Test represents {test_share:.2%} of full dataset""")

    def convert_to_hf_dataset(self) -> None:
        self.huggingface_dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_pandas(self.df.loc[self.train_index]),
                "test": datasets.Dataset.from_pandas(self.df.loc[self.test_index]),
            }
        )
        log.info("Dataset converted to HuggingFace dataset dict.")

    def push_to_huggingface_hub(self):
        self.huggingface_dataset.push_to_hub(
            f"{self.user_name_on_huggingface}/{self.dataset_name_on_huggingface}"
        )
        log.info("Dataset pushed to Hugging Face Hub")

    def __call__(self) -> None:
        self.load_data_in_pandas_df()
        self.perform_train_test_split()
        self.convert_to_hf_dataset()
        self.push_to_huggingface_hub()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create a dataset for SFT of an LLM. Push it to Huggingface Hub"
    )
    subparsers = parser.add_subparsers(
        dest="task",
        help="Create dataset for SFT or push it to Huggingface Hub",
    )
    parser_create_dataset = subparsers.add_parser(
        "create-dataset", help="Prepares and save locally a dataset for SFT"
    )
    parser_create_dataset.add_arguments(PrepareDataConfig, dest="data_config")

    parser_push_to_hub = subparsers.add_parser(
        "push-to-hub", help="Push a clean instruction dataset to hub"
    )
    parser_push_to_hub.add_arguments(PushToHuggingFaceHubConfig, dest="push_config")

    args = parser.parse_args()

    if args.task == "create-dataset":
        prepare_sft_dataset = PrepareSFTDataset(
            dataset_hf_address=args.data_config.dataset_hf_address,
            model_name=args.data_config.model_name,
            prompt_template_name=args.data_config.prompt_template_name,
            local_cache=args.data_config.local_cache,
        )

        prepare_sft_dataset(max_range=args.data_config.max_range)

    elif args.task == "push-to-hub":
        push_dataset_to_huggingface_hub = PushDatasetToHuggingFaceHub(
            local_folder=args.push_config.local_folder,
            user_name_on_huggingface=args.push_config.user_name_on_huggingface,
            dataset_name_on_huggingface=args.push_config.dataset_name_on_huggingface,
            random_seed=args.push_config.random_seed,
            test_split=args.push_config.test_split,
        )

        push_dataset_to_huggingface_hub()


# TODO: chunk article when too big
