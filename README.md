# Replicating Results

## Data Preprocessing

First, we process the raw AMASS data by running the following command.

```bash
python scripts/process_amass_data.py --datasets SFU
```

This will parse all the data files and create sequences of motion data. This will create a training and validation split. To process data and save an output for each file, use the `--by_file` flag.

## Training

To train the model, run the following command. Replace the `data_dir` with the appropriate directory.

```bash
python scripts/train.py --data_dir data/amass_processed/SFU
```

## Testing

To run inference, run the sampling script.

```bash
python scripts/sampling.py --ckpt_path logs/sfu_full/checkpoints/ckpt_epoch=final.pt --data_path data/amass_processed/by_file/0005_Jogging001_stageii.npz
```


## Visualizing

Videos can be made by running the visualization script. This can be done on ground truth data or sampled data from the model.

```bash
python scripts/visualize_amass_data.py --motion_path data/amass_raw/SFU/0005/0005_Jogging001_stageii.npz
```

# TODO

* Autrogressive models with a free-form variance do not generalize because the variance becomes very small around the training data. Use a fixed variance instead.