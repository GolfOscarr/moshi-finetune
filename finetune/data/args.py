import logging
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class DataArgs(Serializable):
    """
     Arguments for data loading. Train and eval data should be jsonl files
    with  "path" and "duration" fields for each audio .wav file.
    """

    train_data: str = ""
    shuffle: bool = False
    eval_data: str = ""
    mode: str = "moshi"

    def __post_init__(self) -> None:
        valid_modes = {"moshi", "hibiki_s2st"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"data.mode must be one of {sorted(valid_modes)}, got {self.mode!r}"
            )
