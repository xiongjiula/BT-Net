# BT-Net
This repo is the official implementation for:\
[BT-Net: A Multi-Scale Transformer U-Net with Morphological Refinement for Adaptive Bone Tissue Segmentation)\
(The details of our BATFormer can be found at the models directory in this repo or in the paper.)

## Usage
### Installation

Check out the official [nnUNet installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/MIC-DKFZ/skeleton-recall.git
cd skeleton-recall
pip install -e .
```


## Requirements
* python >= 3.9
* pytorch 2.2.1
* torchvision 0.17.1

## Datasets
* The Totalsegmentator v2 dataset could be acquired from [here](https://github.com/wasserth/TotalSegmentator).
* The CTSpine1K dataset could be acquired from [here](https://github.com/MIRACLE-Center/CTSpine1K).

## Plan_and Preprocess
Commands for plan and preprocess
``` 
python nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py -d [datasetID] -c [configration] --verify_dataset_integrity
```
## Training
Commands for training
```
python nnunetv2/run/run_training.py [datasetID] [configration] [fold]
```
## Testing
Commands for testing
``` 
python nnunetv2/inference/predict_from_raw_data.py -i [input_path] -o [output_path] -d [datasetID] -c [configration] -f [fold]
```
## Validation
Commands for validation
``` 
python nnunetv2/evaluation/evaluate_predictions.py [labels_dir] [predicts_dir] -djfile [dataset.json_path] -pfile [plans.json_path]
```


## References
1. [nnunet v2](https://github.com/MIC-DKFZ/nnUNet)
2. [BATFormer](https://github.com/xianlin7/BATFormer)

