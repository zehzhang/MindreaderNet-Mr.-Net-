# MindreaderNet (Mr. Net)
This is the repository for our NeurIPS19 paper:

[A Self Validation Network for Object-Level Human Attention Estimation](https://arxiv.org/pdf/1910.14260.pdf)

If you find this repository helpful, consider citing our paper:

```
@inproceedings{attention2019zehua, 
    title = {A Self Validation Network for Object-Level Human Attention Estimation},
    author = {Zehua Zhang and Chen Yu and David Crandall},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2019}
}
```

If any other mentioned work here also helps, please do cite them as well. Thanks!

## Before you start...
This code is a half-cleaned version by removing many parts with which we performed other experiments. We haven't tested it thoroughly because the data we used are removed from the local server. We are re-downloading them and will test the code once the data are ready. But before that, if you find any bugs, please let us know by either issuing a ticket or sending us an email.

## To begin with...
```
cd your_working_directory
git clone https://github.com/zehzhang/MindreaderNet-Mr.-Net-
```

## Environments and Dependencies
Our model is implemented base on [Keras](https://keras.io/) v2.1.5 and [Tensorflow](https://www.tensorflow.org/) v1.10.0

The CUDA version is [9.2](https://developer.nvidia.com/cuda-92-download-archive)

It runs on Linux and we haven't tested it on other OS.

It also uses functions from [ssd_keras](https://github.com/pierluigiferrari/ssd_keras)

```
git clone https://github.com/pierluigiferrari/ssd_keras
```

Make sure ssd_keras is put in the same parent directory as MindreaderNet-Mr.-Net-

## Data Preparation
In our paper two datasets are used: [ATT](http://vision.soic.indiana.edu/papers/gaze2018bmvc.pdf) and [EPIC-Kitchens](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dima_Damen_Scaling_Egocentric_Vision_ECCV_2018_paper.pdf)

In this repository we give an example on running our code on EPIC-Kitchens. 

ATT cannot be made directly available to public due to human ethic and privacy issues. Please contact the authors (also **us**!) for the access to it. 

To download EPIC-Kitchens, use the [script](https://github.com/epic-kitchens/download-scripts/blob/master/frames_rgb_flow/download_rgb.sh) provided by the authors:

```
sh download_rgb your_working_directory/MindreaderNet-Mr.-Net-
```

It will download all the rgb frames to `your_working_directory/MindreaderNet-Mr.-Net-/EPIC_KITCHENS_2018`. Uncompress each .tar file by:

```
python mindreaderv2_clean.py --option extractepic
```

Download the annotations of [EPIC_train_object_labels](https://github.com/epic-kitchens/annotations/blob/master/EPIC_train_object_labels.csv), [EPIC_train_object_action_correspondence](https://github.com/epic-kitchens/annotations/blob/master/EPIC_train_object_action_correspondence.csv) and [EPIC_train_action_labels](https://github.com/epic-kitchens/annotations/blob/master/EPIC_train_action_labels.csv) to `your_working_directory/MindreaderNet-Mr.-Net-` (We've already included them in the repository)

Split the data into two sets for training and testing (This will generated two .txt files --- epic_train.txt and epic_test.txt --- listing sample information. Note that since we choose 90% of the samples randomly, you will get different training and testing lists if you generate them yourself. We've already included the list files we used in the repository):

```
cd your_working_directory/MindreaderNet-Mr.-Net-
python mindreaderv2_clean.py --option processepic
```

Prepare the labels (Four files --- epic_train_onehot.h5, epic_train_labels.h5, epic_test_onehot.h5, epic_test_labels.h5 -- will be generated):

```
python mindreaderv2_clean.py --option encodeepic
```

Load the samples into a big binary file (formatted in .h5) for faster reading during training and testing (Two files will be generated respectively for the training and testing set. By default, both will be put in `your_working_directory/MindreaderNet-Mr.-Net-`. They take several TBs so make sure you have enough disk space.):

```
python mindreaderv2_clean.py --option preloadepic
```

To generate the optical flow maps, we use [PWC](https://arxiv.org/abs/1709.02371). Particularly, we use this [implementation](https://github.com/sniklaus/pytorch-pwc). Flow maps should be generated corresponding to frames stored in the .h5 file and also stored as a .h5 file in the directory of `your_working_directory/MindreaderNet-Mr.-Net-`. For example, the dataset of frames in the training set has shape of (N, H, W, 3), then the dataset of flows in the training set should correspondingly have shape of (N, H, W, 2).

You have the option to download the flow maps generated by the EPIC-Kitchen team with this [script](https://github.com/epic-kitchens/download-scripts/blob/master/frames_rgb_flow/download_flow.sh). We haven't tested with these flows and cannt gurantee the performance.

So far we've done with the data preparation.

For more details about how we process our data, please refer to Section 3.5 of our [paper](https://arxiv.org/pdf/1910.14260.pdf).

## Training

Download the pretrained weights ([I3D](https://arxiv.org/pdf/1705.07750.pdf) is pretrained on [ImageNet](http://www.image-net.org/papers/imagenet_cvpr09.pdf) and [Kinetics](https://arxiv.org/abs/1705.06950), [VGG16](https://arxiv.org/abs/1409.1556) is pretrained on [ImageNet](http://www.image-net.org/papers/imagenet_cvpr09.pdf)): [I3D rgb stream](https://drive.google.com/file/d/1-e6msoYkDkHC0i_a7STqrHvBnMtS5rnE/view?usp=sharing), [I3D flow stream](https://drive.google.com/file/d/1o_mAiMYBveC-jtZFvMlYrKoK4QYvKILx/view?usp=sharing), [VGG16](https://drive.google.com/file/d/1SMarCUF10ykgH-0d7oaZbk4pl9cso84v/view?usp=sharing)

We also provide the weight we obtained by pretraining the spatial branch: [pretrained_spatial_weight](https://drive.google.com/file/d/127LJsKgOQ9N7_0RGVEwIW_U00eB-wn04/view?usp=sharing). Simply download it and put it in `your_working_directory/MindreaderNet-Mr.-Net-`.

To train the model

```
python mindreaderv2_clean.py --option trainepic --batch_size B --steps_per_epoch S --where_help_inside --what_help_inside --softArgmax
```

We train our model on a single Titan Xp with batch size set to 4. You can specify your own batch size by setting ```--batch_size your_batch_size```. Also, the code supports multi-gpu training by setting your own `--num_gpu` (we haven't tested it thoroughly).

When training the model, make sure you specify `--softArgmax` so that the soft verson of what→where validation is used in order to have the gradients backpropagate properly. More detains can be found in Section 3.2 of our [paper](https://arxiv.org/pdf/1910.14260.pdf).

`--where_help_inside` and `--what_help_inside` tell the model whether to use the where→what validation or what→where validation respectively.

By default, a weight file `epic_mindreader.h5` will be saved to `your_working_directory/MindreaderNet-Mr.-Net-`.

Other arguments are also available and please check the code for more details.

## Testing

```
python mindreaderv2_clean.py --option testepic --where_help_inside --what_help_inside --load_exist_model --load_from path_to_your_your_weight
```

Make sure batch_size is set to 1 (or just leave it with its default value of 1).  Only single GPU testing is supported. We also provided the [weight](https://drive.google.com/file/d/1EfWb6xasnkYHd-1CvT3MBSMpfZmyiulh/view?usp=sharing) file we reported in our paper. Download it and put it in `your_working_directory/MindreaderNet-Mr.-Net-`, then
```
python mindreaderv2_clean.py --option testepic --where_help_inside --what_help_inside --load_exist_model --load_from ./epic_acc312.h5
```
