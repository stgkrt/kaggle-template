from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    image_size: int
    num_workers: int
    batch_size: int
    shuffle: bool
    pin_memory: bool


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    num_warmup_steps: int


@dataclass
class SchedulerConfig:
    mode: str
    factor: float
    patience: int


@dataclass
class LossConfig:
    loss_name: str
    pos_weight: float | None


@dataclass
class ModelConfig:
    model_name: str
    backbone_name: str
    pretrained: bool
    in_channels: int
    out_channels: int
    use_batchnorm: bool
    dropout: float
    loss_config: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


@dataclass
class SplitConfig:
    train_study_ids: list[str]
    valid_study_ids: list[str]


@dataclass
class TrainerConfig:
    epochs: int
    accelerator: str
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int


@dataclass
class TrainConfig:
    exp_name: str
    num_epochs: int
    seed: int
    dir: DirConfig
    split: SplitConfig
    model: ModelConfig
    loss: LossConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig
