
from neuralhydrology.utils.config import Config
from neuralhydrology.training.basetrainer import BaseTrainer, LOGGER
from pathlib import Path
from myutils.MyBaseTrainer import MyBaseTrainer

gpu = -1
print(Path('../Static_parameters/base_config.yaml').resolve().absolute())
config = Config(Path('../Static_parameters/base_config.yaml'))

# check if a GPU has been specified as command line argument. If yes, overwrite config
if gpu is not None and gpu >= 0:
    config.device = f"cuda:{gpu}"
if gpu is not None and gpu < 0:
    config.device = "cpu"

# # start training
if config.head.lower() in ['regression', 'gmm', 'umal', 'cmal', '']:
    trainer = MyBaseTrainer(cfg=config)
else:
    raise ValueError(f"Unknown head {config.head}.")

trainer.initialize_training()

# get epoch 0 train loss
trainer._epoch_0()
avg_losses = trainer.experiment_logger.summarise()
loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
LOGGER.info(f"Epoch {0} average loss: {loss_str}")

# get epoch 0 validation loss
if (trainer.validator is not None):
    trainer.validator.evaluate(epoch=0,
                            save_results=trainer.cfg.save_validation_results,
                            save_all_output=trainer.cfg.save_all_output,
                            metrics=trainer.cfg.metrics,
                            model=trainer.model,
                            experiment_logger=trainer.experiment_logger.valid())

    valid_metrics = trainer.experiment_logger.summarise()
    print_msg = f"Epoch 0 average validation loss: {valid_metrics['avg_total_loss']:.5f}"
    # print(print_msg)
    if trainer.cfg.metrics:
        print_msg += f" -- Median validation metrics: "
        print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_total_loss')
        LOGGER.info(print_msg)

trainer.train_and_validate()


