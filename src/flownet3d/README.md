### FlowNet3D: *Learning Scene Flow in 3D Point Clouds*
Created by <a href="http://xingyul.github.io">Xingyu Liu</a>, <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a> and <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from Stanford University and Facebook AI Research (FAIR).

<img src="https://github.com/xingyul/flownet3d/blob/master/doc/teaser.png" width="60%">

### Citation
If you find our work useful in your research, please cite:

        @article{liu:2019:flownet3d,
          title={FlowNet3D: Learning Scene Flow in 3D Point Clouds},
          author={Liu, Xingyu and Qi, Charles R and Guibas, Leonidas J},
          journal={CVPR},
          year={2019}
        }

### Abstract

Many applications in robotics and human-computer interaction can benefit from understanding 3D motion of points in a dynamic environment, widely noted as scene flow. While most previous methods focus on stereo and RGB-D images as input, few try to estimate scene flow directly from point clouds. In this work, we propose a novel deep neural network named FlowNet3D that learns scene flow from point clouds in an end-to-end fashion. Our network simultaneously learns deep hierarchical features of point clouds and flow embeddings that represent point motions, supported by two newly proposed learning layers for point sets. We evaluate the network on both challenging synthetic data from FlyingThings3D and real Lidar scans from KITTI. Trained on synthetic data only, our network successfully generalizes to real scans, outperforming various baselines and showing competitive results to the prior art. We also demonstrate two applications of our scene flow output (scan registration and motion segmentation) to show its potential wide use cases.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.9 GPU version and Python 3.5 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`. It's highly recommended that you have access to GPUs.

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them first by `make` under each ops subfolder (check `Makefile`). Update `arch` in the Makefiles for different <a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported">CUDA Compute Capability</a> that suits your GPU if necessary. The code is tested under TF1.9.0. 

### Usage

#### Flyingthings3d Data preparation

The data preprocessing scripts are included in `data_preprocessing`. To process the raw data, first download <a href="https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html">FlyingThings3D dataset</a>. `flyingthings3d__disparity.tar.bz2`, `flyingthings3d__disparity_change.tar.bz2`, `flyingthings3d__optical_flow.tar.bz2` and `flyingthings3d__frames_finalpass.tar` are needed. Then extract the files in `/path/to/flyingthings3d` such that the directory looks like

```
/path/to/flyingthings3d
  disparity/
  disparity_change/
  optical_flow/
  frames_finalpass/
```

Then `cd` into directory `data_preprocessing` and execute command to generate .npz files of processed data

```
python proc_dataset_gen_point_pairs_color.py --input_dir /path/to/flyingthings3d --output_dir data_processed_maxcut_35_20k_2k_8192
```

The processed data is also provided <a href="https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view?usp=sharing">here</a> for download (total size ~11GB).

#### Training and Evaluation

To train the model, simply execute the shell script `command_train.sh`. Batch size, learning rate etc are adjustable.

```
sh command_train.sh
```

To evaluate the model, simply execute the shell script `command_evaluate.sh`.

```
sh command_evaluate.sh
```

#### KITTI Experiment

To be released. Stay Tuned.

### License
Our code is released under MIT License (see LICENSE file for details).

### Related Projects

* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
* <a href="http://stanford.edu/~rqi/pointnet2" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017). Code and data released in <a href="https://github.com/charlesq34/pointnet2">GitHub</a>.
