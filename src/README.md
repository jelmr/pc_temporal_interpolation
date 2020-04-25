The whole process consist of the following steps:

1. Preprocess training data 
2. Training 
3. Inference
4. Upsampling + snap


## 1. Preprocess training data == 

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
