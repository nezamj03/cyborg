from ...QMix.utils.config import AttributeDict
from functools import partial
from ...QMix.utils.logger import SimpleLogger
from ...QMix.utils.plotting import plot_returns
from .TrainingQMix import QMixTrainer, load_config, load_env
import os

if __name__ == "__main__":

    n_epochs = 1
    returns = []
    for _ in range(n_epochs):
        args = AttributeDict(load_config('CybORG/Evaluation/training_qmix/config/default.yaml'))
        config = args.config
        scheme = args.scheme
        logger = SimpleLogger()
        trainer = QMixTrainer(partial(load_env, seed=42, pad_spaces=True), logger, scheme, config)
        returns.append(trainer.train())

    path = os.path.join("res", "plots")
    os.makedirs(path, exist_ok=True)
    plot_returns(returns, path)
