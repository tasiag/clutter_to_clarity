# Clutter to Clarity
 
To Run: 

(Each of these components can be done separately, as data between each is saved in their respective data folder.)

## Advection-Diffusion Example

### Create Dataset (Diffusion_Convection Folder)

1. Run create_dataset_3D.py (or _2D.py) -- make sure to toggle "SAVE" if you desire to SAVE the output.
    This will create .mat files of the 
    - original dataset (..._original.mat)
    - shuffled dataset (....mat)
    - labels.mat (containing the true orderings of the shuffled data)
    in either the folder 
    - 1_80_100 (for 2D dataset, used for DHPM, PINNS)
    - 64_80_100 (for 3D dataset, used for DeepONet)

### Discover Emergent Coordinates

2. Run questionnaires (download code from https://github.com/RonenTalmonLab/InformedGeometry-CoupledPendulum/tree/main)
   - Load dataset (e.g. 3Dpipe.mat & labels.mat) into Matlab.
   - Run runQuestionnairesPipe.m (this will use setQuestParamsPipe.m)
   - Three embedding figures will load. 
   - Save embeddings.mat to appropriate Output folder.
   - Visualize this data in Python by running "visualize_dataset_{N}D.py". Toggle "SAVE"
     to save out the embeddings and concentrations into nice .npy files (three e_vecs & 1 concentration)

### Create Generative Model
3. Run Neural Networks
    - Choose PINNS (2D data), HiddenPhysics (2D data), or DeepONet (3D data)
    - PINNS: ipynb file in PINNS Folder
    - HiddenPhysics: ipynb file in HiddenPhysics Folder
    - DeepONet: in DeepONet folder, run main_diffusionconvection.py (make_plots_pipe.py following to produce figures. Results will be saved in DeepONet/DiffusionConvection/CaseNo)

## Oscillator Example

### Create Dataset (Oscillators Folder)

1. Run create_dataset.py -- make sure to toggle "SAVE" if you desire to SAVE the output.
    This will create use config.cfg to create .mat files of the 
    - original dataset (3D_data_original.mat)
    - shuffled dataset (3D_data_shuffled.mat or 3D_data_shuffled.npy)
    - data_dict_list.pkl: All details for each trajectory
    - labels.mat (containing the true orderings of the shuffled data)
    - True_K.npy (the K values in order for each simulation run)
    in either the folder specified by the directory in config.cfg.
    There are already two datasets available to play with:
       A) dat0to35_1.08to1.8: This contains a sample run from time = 0 to 35 and K values 1.08 to 1.8. 
       B) dat0to35_1.08to1.8_real: This contains a sample run from time = 0 to 35 and K values 1.08 to 1.8, but only the real part is retained.
    Note that (A) has a different shuffling than (B). 

### Discover Emergent Coordinates

2. Run questionnaires (download code from https://github.com/RonenTalmonLab/InformedGeometry-CoupledPendulum/tree/main)
   - Load dataset (e.g. 3D_data_shuffled.mat & labels.mat) into Matlab.
   - Run runQuestionnairesOscillators.m (this will use setQuestParamsOscillators.m)
   - Three embedding figures will load. 
   - Save embeddings.mat to appropriate Output folder.
   - Post process the embeddings to extract arclengths via postprocessembeddings.py
   - Visualize this data in Python by running "visualize_dataset.py". Toggle "SAVE"
     to save out the embeddings and concentrations into nice .npy files (three e_vecs & 1 concentration)

### Create Generative Model
3. Run Neural Networks
    - DeepONet: in DeepONet folder, run main_oscillators.py (use path variable in dataset_evecs_oscillators.py to select data source)
    - To create figures: run make_plots_oscillator.py. Results will be saved in DeepONet/Oscillators/CaseNo.
    - Ensure data path between make_plots_oscillators and dataset_evecs_oscillators is pointing to the same datasource.
