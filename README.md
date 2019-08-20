# Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks
Paper: Coming soon.

Code: Coming soon.

Data set: Coming soon.

# Visual Results
[Interpolated sequence - Malcolm Samba](https://drive.google.com/open?id=1m3yWUlHyAYU52kWlxmz3BjUIhA5KAnSj)
An example of a dynamic point cloud sequence interpolated using our architecture. For each pair of frames, 5 new frames have been interpolated in between.

[Scene flow visualization - Shae Hip Hop Dancing](https://drive.google.com/open?id=1TXBgubE81hInHW7dQmSpr1djCzc_Kgza) 
This video shows a comparison of the scene flow estimated by our architecture and the ground truth scene flow. On the left: Ours, middle: Ground truth, right: error between ours and ground truth. In the two models on the left (Ours and Ground truth), scene flow direction has been encoded in colors. A brighter or darker color corresponds to larger motion, with the hue of the color being determined by the direction. Gray means no motion. For the model on the right (error), the error between Ours and Ground Truth is shown using a heatmap. Black means no error, with brighter colors meaning a larger error.

# User study videos
The videos used during our user study. Participants were first shown the four videos from the Demonstration list to get familiar with the videos and the viewing interface. Afterwards, they were asked to rate the 32 remaining videos in a randomized order.

**Demonstration**
[GT](https://drive.google.com/open?id=10tBmyN3pfh8GCM3livWKRJFE301Jtm-g), [Low FPS](https://drive.google.com/open?id=1KXm2pIDLIg5t4uO5Ze3M1cMqjHXT_sqZ), [Ours](https://drive.google.com/open?id=1boFfqOJ3zsNEwSOwWW-TXgp54RS7LAAz), [Flownet3D](https://drive.google.com/open?id=1zYNpNj_tfD3NWgCUY08iG559vTh-lyak)

**Rating session**
* Shae Dancing: [GT](https://drive.google.com/open?id=1kpNDl2QJrNpmwYcwPdOSb_Bg_m39kMmr), [Low FPS](https://drive.google.com/open?id=1TfkIL-x1cwXqrpbQZ6gDbVUYdO0xdFRH), [Ours](https://drive.google.com/open?id=1EMgQHRNRv7pzZcJMO60_ra7wywXvxR_Y), [Flownet3D](https://drive.google.com/open?id=1FZ6nbqQpo5hvzqVciYwxO3HKrqk2hc6l)
* Shae Wavedance: [GT](https://drive.google.com/open?id=1ii45UQBSlCkjy5sPesxPtnP1ZMFrR8Nb), [Low FPS](https://drive.google.com/open?id=1i-9P5EoeTqHm3QpaUABrbdqGklU5FKVt), [Ours](https://drive.google.com/open?id=1dZ4Ih2v_rgw010McT8wG-pCWot4CcJ3l), [Flownet3D](https://drive.google.com/open?id=1R2eqWkoxxZfjwKWMui5m-7-apfRBj9qG)
* Shae Hiphopdancing: [GT](https://drive.google.com/open?id=1YKfWINIGnMadp-dSuJ18U6vW6uqkzJE9), [Low FPS](https://drive.google.com/open?id=1RinQCg5PGTGrA_A6nfDT_4Y1ppeYqtm7), [Ours](https://drive.google.com/open?id=1D4Y3zFF_ciZGIZA2-65RRh-eXkbQw3bF), [Flownet3D](https://drive.google.com/open?id=1plhNyKRGk3_bfw8ejdYSptIaatlVUdAl)
* Shae Lookingaround: [GT](https://drive.google.com/open?id=1hpnLIV4cJbk9COSMPZAShsp_eEGOYep5), [Low FPS](https://drive.google.com/open?id=1v0MZu7jfijAq5Ev90Ljdqh7WONAZwWZ1), [Ours](https://drive.google.com/open?id=1t16d6yO5XUQqgkvZFfsQSeOXIENGLj52), [Flownet3D](https://drive.google.com/open?id=1kcXiboQp5rqnhrdjj06atmplS1UYVYB9)
* Malcolm Samba: [GT](https://drive.google.com/open?id=1sowp-kLUiKLhQ04InyyPiZs0xcK-6PX9), [Low FPS](https://drive.google.com/open?id=1CiVqFtsW8ssjk4KsTl2DwFMNtBVjG0XO), [Ours](https://drive.google.com/open?id=1UT7mIcvPaXyMULYyDGtupPb3zoTZM_43), [Flownet3D](https://drive.google.com/open?id=1KFCXyyKMggkOacYlqac1hJFCIlIubksQ)
* Malcolm Fight: [GT](https://drive.google.com/open?id=1rOGd1GcgCr8HBSw1tRWMQDfpsC-YpqHt), [Low FPS](https://drive.google.com/open?id=1vWPKFIavYtnOG-dd77Sz8C5HnFuix5fD), [Ours](https://drive.google.com/open?id=1PcxfgY7XFomxloCzuP2bk21K33cZ7B9z), [Flownet3D](https://drive.google.com/open?id=1jZroqEEtqh-8XyXcLMIGx6bGQOfRG0BO)
* Malcolm Roar: [GT](https://drive.google.com/open?id=1V9CrJUP7p8LiGxfmIEpsnn-OUXjPuCS1), [Low FPS](https://drive.google.com/open?id=1Z270aDm_iWQ_0KtKWELwLCizWkSbcTIt), [Ours](https://drive.google.com/open?id=16njXAwKUS_byu_WT238Bt1T1WNuAJciM), [Flownet3D](https://drive.google.com/open?id=1EvQoLp5PVd2_l55z2t1h-nhCAFjk7zt8)
* Malcolm Yelling: [GT](https://drive.google.com/open?id=1rPy5ZpMY5ZuCbhJylPmeTP92DfvLJsJl), [Low FPS](https://drive.google.com/open?id=1iTsZ-NyxlJjt7Bktyj7uUBeRfQ3hJM6d), [Ours](https://drive.google.com/open?id=1-kJSiTWzrKA_25Hhg6IS8n0aRN9K7Olw), [Flownet3D](https://drive.google.com/open?id=1Hps6MQe5PYPfzbb0fERvfg0cgcArypZE)





