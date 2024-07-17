# Clutter to Clarity
 
To Run: 

(Each of these components can be done separately, as data between each is saved in their respective data folder.)

## Create Dataset

1. Run visualize_dataset_3D.py (or _2D.py) -- make sure to toggle "SAVE" if you desire to SAVE the output.
    This will create .mat files of the 
    - original dataset (..._original.mat)
    - shuffled dataset (....mat)
    - labels.mat (containing the true orderings of the shuffled data)
    in either the folder 
    - 1_80_100 (for 2D dataset)
    - 64_80_100 (for 3D dataset)

## Discover Emergent Coordinates

2. Run questionnaires (download code from https://github.com/RonenTalmonLab/InformedGeometry-CoupledPendulum/tree/main)
   - Load dataset (e.g. 3Dpipe.mat & labels.mat) into Matlab.
   - Run runQuestionnairesPipe.m
   - Three embedding figures will load. 
   - Save embeddings.mat to appropriate Output folder.
   - Visualize this data in Python by running "visualize_dataset_{N}D.py". Toggle "SAVE"
     to save out the embeddings and concentrations into nice .npy files (three e_vecs & 1 concentration)

## Create Forward Model
3. Run Neural Networks
    - Choose PINNS (2D data), HiddenPhysics (2D data), or DeepONet (3D data)
    - PINNS: ipynb file
    - HiddenPhysics: ipynb file
    - DeepONet: run main.py