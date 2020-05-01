The whole process consist of the following steps:

1. Preprocess training data 
2. Training 
3. Inference
4. Upsampling + snap
5. Viewing results
6. Evaluation


## 1. Preprocess training data

Navigate to the `torch` folder.

The config.json file has a bunch of configuration options, one of which is the `data_loader'.'type'`, this should be set to `SyntheticAdvancedDataLoader`, which tells the program to use the data loader defined in `/src/datasets/synthetic_advanced.py`.

This dataloader will use as input the point clouds stored in `data/SyntheticAdvanced/raw`, and save its output to `data/SyntheticAdvanced/processed`. It will only do the preprocessing if the `data/SyntheticAdvanced/processed` folder does not exist yet, so delete it whenever you want process new data. You'll have to then put the point clouds you want to use as input in `data/SyntheticAdvanced/raw` (or symlink it if you prefer). 

Each dynamic point cloud should be stored in one folder. This folder should contain the frames in .ply format, and a file called "frames.dpc", which should be a simple text file listing the names of the frames that are part of that point cloud. You could simply create such a "frames.dpc" file with `ls *.ply >frames.dpc`. Lastly, you'll need to list the names of each directory you want to use in the list returned by `raw_file_names` in `/src/datasets/synthetic_advanced.py`.


For example, for two point clouds the structure might look like this:

```
data
|-SyntheticAdvanced
  |-raw
    |- Jasper_Jumping
        |-frame_001.ply
        |-frame_002.ply
        |-frame_003.ply
        |-frames.dpc
    |- Jasper_SambaDancing
        |-frame_001.ply
        |-frame_002.ply
        |-frames.dpc
```

In this case, the `frames.dpc` file in the `Jasper_Jumping` directory might contain the following three lines:

```
frame_001.ply
frame_002.ply
frame_003.ply
```

The `frames.dpc` file in `Jasper_SambaDancing` might look like this:

```
frame_001.ply
frame_002.ply
```

Of course you don't have to include all the frames.

In this example, you would edit `raw_file_names` in `src/datasets/synthetic_advanced.py` to the following:

```
return [
    "Jasper_SambaDancing",
    "Jasper_Jumping"
]
```

Once this is done, you can run the command `python src/train.py -c config.json`, which should preprocess the data and store the result in `data/SyntheticAdvanced/processed`.

Lastly, edit the paths in `/scripts/split_dataset.py` and then run this script to split `training_dataset.pickle` into a training- and test set.


## 2. Training 
Navigate to the `flownet3d` folder.

The dataset used for training is defined in `synthetic_dataset.py`, you should edit the two paths in the constructor to point to the two files you created with the `split_dataset.py` script.

```
if self.train:
    self.root = '/path/to/training_dataset_train.pickle'
else:
    self.root = '/path/to/training_dataset_test.pickle'
```

The training phase consists of two parts (see Thesis Section 3.4).

Training the first part:
```
python train.py
```

This starts training and by default will write logs and the trained weights to `/log_train` (this can be modified with the `--log_dir` flag).

Training the second part:
```
python train_part2.py
```

This takes the weights from the first part of training (assumed to be in `log_train`, if you stored them elsewhere, pass that directory using the `--model-path` flag). This script will perform the second part of training and output logs and weights to `log_train__phase2` (can be modified using the `--log_dir` flag, again).

You can monitor training using Tensorboard:
```
tensorboard --logdir <path_to_the_logdir> --port <port_nr>
```
This starts an HTTP server on <port_nr>.

## 3. Inference
To perform the interpolation, we'll need both a low resolution (lr) version of each frame (consisting of 2048 points), and a high resolution version (hr), consisting of an arbitrary number of points. In this step we will use the LR version as input to the network, which will then output a LR point cloud with estimated scene flow. In the next step we will upsample this result.

```
python interpolate_full.py --in_dir /path/to/lr_input_dpc --in_dpc /path/to/lr_input_dpc/frames.dpc --out_dir /path/to/lr_output_dpc --out_dpc /path/to/lr_output_dpc/frames.dpc
```

Additionally, you should use the `--model_path` flag you didn't use the default directory (`log_train_phase2`) for `train_part2.py`.

This will output the interpolated point clouds to the provided output directory.

## 4. Upsampling + snap
In this step we will upsample the LR point cloud from the previous snap, and apply point snapping to make the interpolation result smoother.

```
python upsample_and_snap.py /path/to/lr_input_dpc /path/to/lr_input_dpc/frames.dpc /path/to/hr_input_dpc /path/to/hr_input_dpc/frames.dpc /path/to/output_dpc
```

[There is also a `batch_upsample_and_snap.py` script, that will automatically run this for multiple dpcs, but you might have to modifiy it a bit for your needs.]

Here:
- `lr_input_dpc` is the output created by `interpolate_full.py` in the previous step
- `hr_input_dpc` is the high resolution dpc corresponding to the low resolution point cloud you used as input for `interpolate_full.py`.
- `output_dpc` is where the result will be stored.

The output of this step is a high resolution point cloud, with an estimated sceneflow for each point. This estimated sceneflow can be used in the next step to render the interpolation.

## 5. Viewing results
In the `scripts` folder is a `dynamic_point_cloud_renderer.py` which you can use to render the interpolation.

```
python dynamic_point_cloud_renderer.py /path/to/dpc /path/to/dpc/frames.dpc
```

Run with the `--help` flag to see all options. Most important options are `--fps n`, to set the frames per second before interpolation, and `--interpolate n`, to interpolate `n` frames between each frame pair (based on the estimated sceneflow).

Press `q` to start the animation, use `-` and `=` to increase the point size.


## 6. Evaluation

The `calculate_objective_metrics.py` script can be used to calculate objective metrics for an interpolation result.

It takes:
- `--in_dir`: The directory containing the interpolation result (expected to have a `frames.dpc` file in it)
- `--gt_dir`: The directory containing the ground truth. This directory should contain a `sceneflow` directory, which contains a DPC with sceneflow, and a `normals` directory, which contains a DPC with normals.
- `out`: Directory where to write the results.




