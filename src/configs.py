from dataclasses import dataclass


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str


@dataclass
class DatasetConfig:
    _target_: str
    data_dir: str
    train_num: int
    valid_num: int
    test_num: int
    batch_size: int
    num_workers: int
    pin_memory: bool


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
    _target_: str
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
    seed: int
    ckpt_path: str
    dir: DirConfig
    model: ModelConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
