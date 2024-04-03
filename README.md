# Amino Acid Sequence Classification using BERT

This repository contains a Python script, `FineTune.py`, which utilizes the BERT (Bidirectional Encoder Representations from Transformers) model to classify amino acid sequences into Antimicrobial Peptide (AMP) or non-AMP sequences. The script fine-tunes a pre-trained BERT model for sequence classification.

## Dependencies

Before running the script, ensure you have the following dependencies installed:

- pandas
- scikit-learn
- transformers
- torch
- matplotlib
- numpy
- tqdm

You can install these dependencies using pip:

```
pip install pandas scikit-learn transformers torch matplotlib numpy tqdm
```

## Usage

1. Clone this repository to your local machine:
```
git clone https://github.com/gmausa/Peptide_Transformer.git
```
2. Navigate to the repository directory:
```
cd Peptide_Transformer
```

3. Place your dataset file named `amp_dataset.csv` in the repository directory.

4. Run the `FineTune.py` script:
```
python FineTune.py
```

## Dataset

The dataset (`amp_dataset.csv`) should contain two columns:

- `sequence`: Amino acid sequences.
- `label`: Binary labels indicating whether the sequence is an AMP (1) or non-AMP (0).

## Training

The script splits the dataset into training, validation, and test sets. It then fine-tunes the BERT model on the training data and evaluates the model's performance on the validation set. Training progress and validation metrics are displayed during the training process.

## Evaluation

After training, the script evaluates the fine-tuned model on the test set and reports various evaluation metrics, including accuracy, precision, recall, F1 score, and ROC AUC score.

## Model Saving

The fine-tuned model is saved to `Models/model_file.bin` after training. If the model file already exists, the script loads the pre-trained model from the file instead of retraining.

## Note

- Make sure to adjust the file paths and hyperparameters in the script according to your requirements.
- You may need to modify the script if your dataset or task has specific characteristics.
- Feel free to experiment with different BERT models, hyperparameters, and optimization strategies to improve performance.

Please refer to the script comments for detailed explanations of each step and customization options. If you have any questions or suggestions, feel free to open an issue or contact the repository owner. Happy coding!

