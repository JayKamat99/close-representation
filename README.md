# CloSE: A Compact Shape- and Orientation-Agnostic Cloth State Representation

### This repository contains the code for the paper submitted to IROS 2025

<div align="center">
  <video src="./media/visualization/half_tshirt_01.mp4" autoplay controls muted loop playsinline width="30%"></video>
  <video src="./media/visualization/75half_disk_01.mp4" autoplay controls muted loop playsinline width="30%"></video>
  <video src="./media/visualization/50diagonalHalf_napkin_01.mp4" autoplay controls muted loop playsinline width="30%"></video>
</div>

## Getting Started
First clone the repository and move into it
```
git clone https://github.com/JayKamat99/close-representation.git
cd close-representation
```

Create a conda environment and install all the necessary packages
```
conda create --name CloSE python=3.9
conda activate CloSE
pip install -e .
```

## Download Dataset
We have created a small cloth folding dataset using MATLAB which can be downloaded from [here](https://drive.google.com/drive/folders/1Cnfgw4cZaeS2bdJCLvTXuPjivlGwuTWo?usp=sharing)
The code assumes that the dataset is stored in the folder Dataset

## Examples
You may run the examples in the examples folder using the command. Specific examples below.
```
python Examples/example.py --filePath="relative/path/to/file"
```
If you do not specify any file path, it will evaluate on a default example

### Visualizing the dGLI disk
This generates the videos shown above. chaneg the filePath to visualize different examples
```
python Examples/visualize_dGLI_disk.py --filePath="Dataset/dataset_MATLAB/50diagonalHalf_napkin_01.mat"
```
This will create an animation of how the dGLI disk evolves with folds. The animation will be saved in the folder `Results`

### CloSE representation is continuous
<!-- Put gifs here like done at the start. Maybe the start can be replaced by the pull figure -->
<div align="center">
  <video src="./media/continuity/Cloth_1.mp4" autoplay controls muted loop playsinline width="30%"></video>
  <video src="./media/continuity/Tshirt_3.mp4" autoplay controls muted loop playsinline width="30%"></video>
  <video src="./media/continuity/Tshirt_1.mp4" autoplay controls muted loop playsinline width="30%"></video>
</div>

```
python Examples/visualize_continuity.py
```
Here you may add flags: `--filePath="relative/path/to/file"` for the data location and optionally add the desired start and the end fold locations for intepolation by using  the flags `--start=[a,b]` and `--end=[c,d]`
For example:
```
python Examples/visualize_continuity.py --filePath="Dataset/dataset_MATLAB/50diagonalHalf_napkin_01.mat" --start=[0,2.9] --end=[1.1,2.1]
```
The resaults are saves in the folder `Results`

### Semantic Labelling
```
python Examples/get_semantic_label.py --filePath="Dataset/dataset_MATLAB/bottomTopHalf_Tshirt_03.mat"
```

### Planning
```
python Examples/get_plan.py
```
optionally, you can add the `--filePath` and The goal fold location with `--goal=[f_1,f_2]` where the array [$f_1$, $f_2$] indicates the goal fold location on the CloSE representation.

For more information refer to the paper website at [close-representation](https://jaykamat.me/close-representation)