import random
from collections import defaultdict
from typing import Iterator, List, Optional

from torch.utils.data import Sampler
from transformers import TrainerCallback
from trl import SFTTrainer


class LanguageGroupedSampler(Sampler):
    """
    Sampler that groups samples by language to create homogeneous batches.
    Each batch will contain samples from only one language.
    """

    def __init__(
        self,
        languages: List[str],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group indices by language
        self.language_to_indices = defaultdict(list)
        for idx, lang in enumerate(languages):
            self.language_to_indices[lang].append(idx)

        self.languages = list(self.language_to_indices.keys())
        print(f"Languages found: { {l: len(idxs) for l, idxs in self.language_to_indices.items()} }")

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)

        # Shuffle within each language group
        language_indices = {}
        for lang, indices in self.language_to_indices.items():
            idx_copy = indices.copy()
            if self.shuffle:
                rng.shuffle(idx_copy)
            language_indices[lang] = idx_copy

        # Build batches: each batch is mono-lingual
        all_batches = []
        for lang, indices in language_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                # Drop last incomplete batch
                if len(batch) == self.batch_size:
                    all_batches.append(batch)

        # Shuffle the order of batches across languages
        if self.shuffle:
            rng.shuffle(all_batches)

        # Flatten batches into a sequence of indices
        for batch in all_batches:
            yield from batch

    def __len__(self) -> int:
        total = sum(
            (len(v) // self.batch_size) * self.batch_size
            for v in self.language_to_indices.values()
        )
        return total

    def set_epoch(self, epoch: int):
        """Call this each epoch to get different shuffling."""
        self.epoch = epoch


class LanguageGroupedSFTTrainer(SFTTrainer):
    """SFTTrainer with homogeneous language batching."""

    def __init__(self, *args, languages: List[str], **kwargs):
        self.languages = languages
        self._language_sampler = None  # set in _get_train_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        sampler = LanguageGroupedSampler(
            languages=self.languages,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            seed=self.args.seed,
        )
        self._language_sampler = sampler  # store reference for the callback
        return sampler


class EpochShuffleCallback(TrainerCallback):
    """Updates the sampler seed at the start of each epoch so shuffling varies."""

    def __init__(self, trainer: LanguageGroupedSFTTrainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.trainer._language_sampler is not None:
            self.trainer._language_sampler.set_epoch(int(state.epoch))