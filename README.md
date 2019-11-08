# Lyft 3D Object Detection for Autonomous Vehicles
EC601 Project

## Requirements
* Python >= 3.7
* PyTorch >= 1.0
* CUDA 10 and cudnn 7
The build processes have only be tested on Ubuntu 18.04 and 19.10

## Clone the repository
```bash
git clone git@github.com:YUHZ-ACA/Lyft-3D-Object-Detection.git --recursive
```

## Build and Install
Before training and evaluation, we have to build and install some binaries first.
### spconv
Following the instruction in [spconv](https://github.com/traveller59/spconv):
1. `git clone https://github.com/traveller59/spconv --recursive`
2. install `libboost`: by package manager (Ubuntu package name `libboost-all-dev`) or download from official site then put headers into `spconv/include`
3. make sure `cmake` >= 3.13.2 and the executable in `PATH`
4. navigate into `spconv` directory, then run `python setup.py bdist_wheel`
5. `cd ./dist`, then install the wheels by `pip install [WHEELS_NAME]` or `pip3 install [WHEELS_NAME]`

##### Common Build Problems and Possible Solutions
1. CUDA not found

Ensure CUDA is installed with proper nvidia graphics driver. Ensure that cudnn has been installed correctly.

2. `/usr/local/cuda/lib64/xxx.so` is required by xxx file

If your CUDA installation is not placed in `/usr/local`, e.g. directly `apt install nvidia-cuda-toolkit`, please figure out the directory of `bin` and `lib64`, and soft link the installation directory to target directory (you can use `sudo ln -s [LINK_NAME]`).

### SECOND
1. Install all dependencies (for `conda` user)
```bash
conda install scikit-image scipy numba pillow matplotlib
pip install fire tensorboardX protobuf opencv-python nuscenes-devkit lyft_dataset_sdk
```

2. Install SECOND (for `conda` user)
At top level of SECOND, use
```bash
conda develop .
```

## Train and Evaluation

### Data Preparation


### Train SECOND / PointPillars



## Utils
Use `utils/convert_to_kitti.py` to convert the Dataset from nuScene (Lyft/Kaggle) to KITTI.

*Note that `kitti.py` and `export_kitti.py` are based on the code in [lyft_dataset_sdk](https://github.com/lyft/nuscenes-devkit)*

## Sprint 1
Please check out the [slides](./docs/sprint1_slides.pdf)

