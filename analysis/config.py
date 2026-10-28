"""Strict TOML configuration; relative paths are resolved from the repository root."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    name: str = 'paper'
    input_dir: str = 'data/wired'
    output_dir: str = 'outputs/paper'
    model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    revision: str = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
    local_files_only: bool = True
    batch_size: int = 32
    seed: int = 20260913
    centroid_min_words: int = 5
    bootstrap_replicates: int = 999
    alpha: float = 0.05

    def __post_init__(self):
        for key, minimum in [('batch_size', 1), ('seed', 0), ('centroid_min_words', 0), ('bootstrap_replicates', 1)]:
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        if type(self.local_files_only) is not bool:
            raise ValueError('local_files_only must be true or false')
        if not isinstance(self.alpha, (float, int)) or not 0 < self.alpha < 1:
            raise ValueError('alpha must be between zero and one')
        for key in ['name', 'model', 'revision', 'input_dir', 'output_dir']:
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f'{key} must be a nonempty string')
        if self.output == ROOT or self.output == self.input or self.output in self.input.parents or self.input in self.output.parents:
            raise ValueError('Output directory must be separate from input data and repository root')

    @property
    def input(self):
        return (ROOT / self.input_dir).resolve()

    @property
    def output(self):
        return (ROOT / self.output_dir).resolve()

    @property
    def selection(self):
        return 'all nonempty utterances' if self.centroid_min_words == 0 else f'n_words >= {self.centroid_min_words}'

    def to_dict(self):
        return asdict(self)


def load_config(path=None, output_dir=None):
    path = Path(path) if path else ROOT / 'analysis/configs/paper.toml'
    config = Config(**tomllib.loads(path.read_text()))
    return replace(config, output_dir=str(output_dir)) if output_dir else config
