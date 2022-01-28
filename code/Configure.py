# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
    "save_dir": '../model_checkpoints/',
    "depth": 2,
    "num_classes": 10,
    "model_version": 1,
    "first_num_filters": 32,
    "weight_decay": 1e-5,
    "learning_rate": 0.1,
    "lr_decay": 1e-4,
    "momentum": 0.7,
    "max_epochs": 100,
}

training_configs = {
    "batch_size": 256,
    "save_interval": 10,
    # ...
}

### END CODE HERE