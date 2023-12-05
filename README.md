# Replicating Results

## Data Preprocessing

First, we process the raw AMASS data by running the following command.

```bash
python scripts/process_amass_data.py --datasets SFU
```

This will parse all the data files and create sequences of motion data.

## Training

To train the model, run the following command. Replace the `data_dir` with the appropriate directory.

```bash
python scripts/train.py --data_dir data/amass_processed/SFU
```

## Visualizing

Videos can be made by running the visualization script. This can be done on ground truth data or sampled data from the model.

```bash
python scripts/visualize_amass_data.py --motion_path data/amass_raw/SFU/0005/0005_Jogging001_stageii.npz
```

# TODO

* Autrogressive models with a free-form variance do not generalize because the variance becomes very small around the training data. Use a fixed variance instead.