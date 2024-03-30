from abc import abstractmethod
from pathlib import Path
from typing import Protocol, Union

from datasets import DatasetDict, load_dataset, load_from_disk


class ClimateSFTDatasetRepository(Protocol):
    path: Union[Path, str]

    @abstractmethod
    def save(self, dataset_dict: DatasetDict) -> None:
        pass

    @abstractmethod
    def load(self) -> DatasetDict:
        pass


class TemplatedClimateSFTDatasetRepository(Protocol):
    path: Union[Path, str]

    @abstractmethod
    def save(self, dataset_dict: DatasetDict) -> None:
        pass

    @abstractmethod
    def load(self) -> DatasetDict:
        pass


class HubClimateSFTDatasetRepository(ClimateSFTDatasetRepository):
    path = "pierre-pessarossi/climate-question-answers"

    def save(self, dataset_dict: DatasetDict) -> None:
        dataset_dict.push_to_hub(self.path)

    def load(self) -> DatasetDict:
        return load_dataset(self.path)


class LocalTemplatedSFTDatasetRepository(TemplatedClimateSFTDatasetRepository):
    path = Path("climateGPT/data/templated_sft_dataset")

    def save(self, dataset_dict: DatasetDict) -> None:
        dataset_dict.save_to_disk(self.path)

    def load(self) -> DatasetDict:
        return load_from_disk(self.path)
