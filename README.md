# How Do You Perceive My Face? Recognizing Facial Expressions in Multi‐Modal Context by Modeling Mental Representations

## Requirements
- Clone the repository
    ``` bash
  git clone [todo]
    ```
- Create anaconda enviroment follow

    ``` bash
    cd context_matters
    conda env create -f environment.yaml 
    ```
- Download the processed data under the links [RAVDESS](https://drive.google.com/file/d/134RVU8PFh35qWOp7jleIY9i44wG0FZFy/view?usp=sharing), [mead](https://wywu.github.io/projects/MEAD/MEAD.html)
- For evaluation and test purposes, we provide pre-trained model on [RAVDESS](https://drive.google.com/file/d/1jstgr7WRlvp1WhkgIle707yGEoxXjFOZ/view?usp=sharing) and [mead](https://drive.google.com/file/d/19G8df5lcfscYrNZ1gP2NtiBqCx-6OddN/view?usp=sharing)
## Training
The training of the entire framework contains 3 steps
### 1. VAE face and audio spectrogram reconstruction training
Edit the data_path with the path of your dataset and the in_channel in ```Residual_vaegan.yaml``` according to your input data and run
```bash
python run.py --config configs/Residual_vaegan.yaml --command fit
```
### 2. Initial facial expression classifier training
Modify the ckpt in ```MLP_classify.yaml``` with the path of face reconstruction model you obtained in step 1 and run the following command
```bash
python run.py --config configs/MLP_classify.yaml --command fit
```
### 3. CAN network training
Compile the ckpts of backbone_1, backbone_2 and classifier in ```ia_attention_class.yaml``` with the models you trained in the last two steps. In the end run the command
```bash
python run.py --config configs/ia_attention_class.yaml --command fit
```
## Testing
Edit the ckpt in ```ia_attention_class.yaml``` with the model you trained or the one we provide, and execute the command 
```bash
python run.py --config configs/ia_attention_class.yaml --command eval
```
It will print the validation accuracy, followed by the test accuracy.

## Image Generation
You can select one image and an audio file from the provided dataset and produce
the merged face image using the following command
```bash
python interface.py --image path/to/image --audio path/to/audio
```