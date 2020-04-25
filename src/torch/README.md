# PyTorch Template Project
PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch Template Project](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss and metrics](#loss-and-metrics)
			* [Multiple metrics](#multiple-metrics)
		* [Additional logging](#additional-logging)
		* [Validation datasets](#validation-data)
		* [Checkpoints](#checkpoints)
    * [TensorboardX Visualization](#tensorboardx-visualization)
	* [Contributing](#contributing)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5
* PyTorch >= 0.4
* tqdm (Optional for `test.py`)
* tensorboard >= 1.7.0 (Optional for TensorboardX)
* tensorboardX >= 1.2 (Optional for TensorboardX)

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, datasets shuffling, and validation datasets splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── config.json - config file
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for datasets loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── datasets/ - anything about datasets loading goes here
  │   └── data_loaders.py
  │
  ├── datasets/ - default directory for storing input datasets
  │
  ├── model/ - models, losses, and metrics
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── saved/ - default checkpoints folder
  │   └── runs/ - default logdir for tensorboardX
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      ├── logger.py - class for train logging
      ├── visualization.py - class for tensorboardX visualization support
      └── ...
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python3 train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "MnistModel",       // name of model architecture to train
    "args": {

    }                
  },
  "datasets": {
    "type": "MnistDataLoader",         datasets
    datasetss":{
      "data_dir": datasets,        datasets // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 datasets
      "validation_split": datasets         datasets
      "num_workers": 2,                datasets
    }datasets,
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                  datasets/ learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
    "my_metric", "my_metric2"          // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                   // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboardX": true,              // enable tensorboardX visualization support
    "log_dir": "saved/runs"            // directory to save log files for visualization
  }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization
### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.Dadatasetsader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    *datasetsa shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is datasetsterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data:
      pass
  ```
* **Example**

  Please refer to `data/data_loaders.py` for an MNIST data loading exampdatasets
### Trainer
* **Writing your own trainer**

1. **Inhdatasets ```BaseTrainer```**

    `Basedatasetsner` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

#### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["my_metric", "my_metric2"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log = {**log, **additional_log}
  return log
  ```
  
### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, it will return a vdatasetsation data loader, widatasetshe numbedatasets samples according to the specified ratio in your config file.

**Note**: the `datasetst_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkdatasetsts
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### TensorboardX Visualization
This template supports [TensorboardX](https://github.com/lanpa/tensorboardX) visualization.
* **TensorboardX Usage**

1. **Install**

    Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Set `tensorboardX` option in config file true.

3. **Open tensorboard server** 

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in thisdatasetsplate are basically wrappers for those of `tensorboardX.SummaryWriter` module. 

**Note**: You don't have to specify current steps, since `WriterTensorboardX` class defined at `logger/visualization.py` will track current steps.

## Contributing
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs
- [ ] Iteration-based training (instead of epoch-based)
- [ ] Multiple optimizers
- [ ] Configurable logging layout, checkpoint naming
- [ ] `visdom` logger support
- [x] `tensorboardX` logger support
- [x] Adding command line option for fine-tuning
- [x] Multi-GPU support
- [x] Update the example to PyTorch 0.4
- [x] Learning rate scheduler
- [x] Deprecate `BaseDataLoader`, use `torch.utils.data` instesad
- [x] Load settings from `config` files

## License
This project is licensed under the MIT Licedatasets See  LICENSE for more details

## Acknowledgments
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
