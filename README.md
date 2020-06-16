
# Leveraging 2D Data to Learn Textured 3D Mesh Generation

This repository contains the original implementation of the above [CVPR paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Henderson_Leveraging_2D_Data_to_Learn_Textured_3D_Mesh_Generation_CVPR_2020_paper.html) (also available on [arXiv](https://arxiv.org/abs/2004.04180)).
It is a structured VAE that learns a distribution of textured 3D shapes from just 2D images, 
by learning to explain those images in terms of 3D shapes differentiably rendered over a 
2D background.
It also includes a mesh parameterisation guaranteed to avoid self-intersections, by having faces 
push each other out of the way when the shape is deformed.

If this code is useful for your research, please cite us!
```
@inproceedings{henderson20cvpr,
  title={Leveraging {2D} Data to Learn Textured {3D} Mesh Generation},
  author={Paul Henderson and Vagia Tsiminaki and Christoph Lampert},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```


## Prerequisites

- Clone this repo recursively, with
```
git clone --recursive https://github.com/pmh47/textured-mesh-gen
```
- Create and activate a new conda environment, then run
```
conda install python=3.6 tensorflow-gpu=1.13.1 numpy scipy opencv ffmpeg tqdm
pip install tensorflow-probability==0.6 trimesh meshzoo
pip install --no-deps git+https://github.com/pmh47/dirt
```


## Pushing Mesh Parameterisation

The LP-based mesh-pushing op is independent of the rest of the code, and can be found in `src/mesh_intersections`.
It must be compiled before use.


### Compilation

- Ensure you have the system packages Boost, GMP, MPFR, GLFW, and CMake (3.14 or newer) installed
- Install [Gurobi](https://www.gurobi.com/products/gurobi-optimizer/) (and get a licence for it); we used version 8.1.1. 
Set the environment variable `GUROBI_ROOT` to the path containing its include and lib folders, e.g.
```
export GUROBI_ROOT=~/packages/gurobi811/linux64
```
- Activate a conda env with the packages listed under prerequisites
- Run
```
cd src/mesh_intersections
mkdir build && cd build
cmake \
    -DLIBIGL_WITH_CGAL=ON \
    -DLIBIGL_WITH_OPENGL=ON \
    -DLIBIGL_WITH_OPENGL_GLFW=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGUROBI_ROOT=$GUROBI_ROOT
    ..
make -j4
cd ..
```
- Test the C++ code: run `./test_app`, which should display an icosphere with two deformations applied (press ESC to close each window)
- Test the python bindings: run `PYTHONPATH=.. python test_tf_mesh_pushing.py`


### Troubleshooting

You might need to make some of the following changes, if the build fails or you experience crashes:
- Replace `libigl/external/eigen` with a symlink to the version in your tensorflow-gpu package
- Patch tensorflow's eigen following https://bitbucket.org/eigen/eigen/commits/88fc23324517/
- Patch tensorflow's absl following https://github.com/tensorflow/tensorflow/issues/31568#issuecomment-547198495 (causes a crash at runtime, missing a `basic_string_view` symbol)

Note that it is normal to see occasional warnings about numerical issues at runtime.


## Training


We include the full implementation of the structured VAE model described in the paper, as well
as code for preprocessing datasets.
If you want to use the mesh-pushing parameterisation, follow the steps above first. However, it is quicker and 
simpler to begin by training with the **dense** parameterisation.

### Datasets

Before training the model, you need to either download one of our preprocessed datasets, or generate this 
yourself.
Preprocessed datasets are available at the following locations:
- [BrnoCompSpeed](https://pub.ist.ac.at/~phenders/textured-mesh-gen/BrnoCompSpeed.zip)
- [CUB-200-2011](https://pub.ist.ac.at/~phenders/textured-mesh-gen/CUB-200-2011.zip)
- [ShapeNet (HSP)](https://pub.ist.ac.at/~phenders/textured-mesh-gen/HSP.zip)
- [ShapeNet (3D-R2N2)](https://pub.ist.ac.at/~phenders/textured-mesh-gen/3D-R2N2.zip)

Each contains a single folder, which should be unzipped into the `preprocessed-data` folder in this repo.

If you prefer to preprocess the data yourself, we include scripts `extract_*_crops.py` for doing so.
Note that to preprocess the BrnoCompSpeed data, you will need [Detectron](https://github.com/facebookresearch/Detectron) installed.
 You should download the original datasets from the following locations:
- BrnoCompSpeed -- you need `2016-ITS-BrnoCompSpeed-full.tar` from [here](https://medusa.fit.vutbr.cz/traffic/research-topics/traffic-camera-calibration/brnocompspeed/)
- CUB-200-2011 -- you need the [raw dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and [CMR 's `cachedir.tar.gz`](https://github.com/akanazawa/cmr/blob/master/doc/train.md#cub-data)
- ShapeNet (HSP) -- you need `shapenet_data.tar.gz` from [here](https://github.com/chaene/hsp#training-network)
- ShapeNet (3D-R2N2) -- you need `ShapeNetRendering.tar.gz` from [here](https://github.com/chrischoy/3D-R2N2#datasets)

These should be unzipped into the `data` folder in this repo before running the relevant preprocessing script.


### Training

The following commands may be run from the `src` folder to reproduce (up to stochastic variability and small
bug-fixes) the models used in the paper, in setting **mask** with parameterisation **dense**.
Images will be written to `output/images` at regular intervals.
- BrnoCompSpeed: `python train.py dataset=bcs`
- CUB-200-2011: `python train.py dataset=cub`
- ShapeNet cars (3D-R2N2): `python train.py dataset=shapenet synset=02958343`
- ShapeNet chairs (HSP): `python train.py dataset=shapenet synset=03001627 `
- ShapeNet aeroplanes (HSP): `python train.py dataset=shapenet synset=02691156`
- ShapeNet sofas (3D-R2N2): `python train.py dataset=shapenet synset=04256520`

To use setting **no-mask** instead, add `with-gt-masks=0` to any of the above.
To use parameterisation **pushing** instead, add `shape-model=VAE-seq-att-pushing`.
