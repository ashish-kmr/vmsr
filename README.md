# Learning Navigation Subroutines from Egocentric Videos
**Ashish Kumar, Saurabh Gupta, Jitendra Malik**

**Conference on Robot Learning (Corl) 2019.**

**[ArXiv](https://arxiv.org/abs/1905.12612), 
[Project Website](https://ashishkumar1993.github.io/subroutines/)**

### Citing
If you find this code base and models useful in your research, please consider
citing the following paper:

```
@article{kumar2019learning,
  title={Learning navigation subroutines by watching videos},
  author={Kumar, Ashish and Gupta, Saurabh and Malik, Jitendra},
  journal={arXiv preprint arXiv:1905.12612},
  year={2019}
}
```
### Contents
1.  [Requirements: software](#requirements-software)
2.  [Requirements: data](#requirements-data)
3.  [Train and Evaluate Models](#train-and-evaluate-models)


### Requirements: software
1.  Python Virtual Env Setup: All code is implemented in Python but depends on a
    small number of python packages and a couple of C libraries. We recommend
    using virtual environment for installing these python packages and python
    bindings for these C libraries.
      ```Shell
      VENV_DIR=venv
      pip install virtualenv
      virtualenv $VENV_DIR
      source $VENV_DIR/bin/activate
      
      # You may need to upgrade pip for installing openv-python.
      pip install --upgrade pip
      # Install simple dependencies.
      pip install -r requirements.txt

      # Patch bugs in dependencies.
      sh patches/apply_patches.sh
      ```

2.  Swiftshader: We use
    [Swiftshader](https://github.com/google/swiftshader.git), a CPU based
    renderer to render the meshes.  It is possible to use other renderers,
    replace `SwiftshaderRenderer` in `render/swiftshader_renderer.py` with
    bindings to your renderer. 
    ```Shell
    mkdir -p deps
    git clone --recursive https://github.com/google/swiftshader.git deps/swiftshader-src
    cd deps/swiftshader-src && git checkout 91da6b00584afd7dcaed66da88e2b617429b3950
    wget https://chromium.googlesource.com/native_client/pnacl-subzero/+archive/a018d6e2dc9b3f0b1a48d1deade8160e44589189.tar.gz
    tar xvfz a018d6e2dc9b3f0b1a48d1deade8160e44589189.tar.gz -C third_party/pnacl-subzero/
    mkdir build && cd build && cmake .. && make -j 16 libEGL libGLESv2
    cd ../../../
    cp deps/swiftshader-src/build/libEGL* libEGL.so.1
    cp deps/swiftshader-src/build/libGLESv2* libGLESv2.so.2
    ```

3.  PyAssimp: We use [PyAssimp](https://github.com/assimp/assimp.git) to load
    meshes.  It is possible to use other libraries to load meshes, replace
    `Shape` `render/swiftshader_renderer.py` with bindings to your library for
    loading meshes. 
    ```Shell
    mkdir -p deps
    apt-get install libxext-dev libx11-dev
    git clone https://github.com/assimp/assimp.git deps/assimp-src
    cd deps/assimp-src
    git checkout 2afeddd5cb63d14bc77b53740b38a54a97d94ee8
    cmake CMakeLists.txt -G 'Unix Makefiles' && make -j 16
    cd port/PyAssimp && python setup.py install
    cd ../../../..
    cp deps/assimp-src/lib/libassimp* .
    ```

### Requirements: data
1.  Download the Stanford 3D Indoor Spaces Dataset and Matterport 3D dataset. 
    The expected data format is ../data/mp3d/<meshes/class-maps/room-dimension>, where the mp3d 
    folder contains the data from both MP3D and S3DIS.

### Train and Evaluate Models
You can download the pretrained models by running the script `scripts/download_pretrained_models.sh`. 

1. To train an inverse model, run the script `scripts/train_inverse_model.sh` (pretrained model included in the download script).

2. To train a vmsr policy, run the script `scripts/train_vmsr.sh` (pretrained model included in the download script). 

3. To evaluate the pretrained model on exploration, run the script `scripts/evaluate_exploration.sh`.

4. To evaluate the pretrained model on downstream RL for point goal and area goal, run the script `scripts/downstream_rl.sh`. This script contains 4 settings corresponding to area goal and point goal tasks, each for dense and sparse rewards.

### Credits
This code was written by Ashish Kumar and Saurabh Gupta. The rl code was adapted from [PyTorch Implementations of Reinforcement Learning Algorithms](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) Kostrikov, Ilya; GitHub. 2018
