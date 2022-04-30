# IFT6759
Advanced Machine Learning Projects

Create the environment:
    conda env create --file environment.yml
	
Activate the environment:
    conda activate ift6759-env

Update the environment:
	conda env update --name ift6759-env --file environment.yml --prune



General Archetecture:

`main.py` reads the configuration from the `.yaml` file located in `Config` folder and carries out the instructions.
The `Data` folder contains code that retrieves the data. The retrieved data are stored in the corresponding subfolder. (Default: Cifar10)
The `Augmentation` folder contains code for performing data augmentation. The augmented data are saved in the corresponding subfolder. 
The `Model` folder contains code for generating the desired model structure. The performance parameters are saved in the corresponding text file. (Default: Resnet20)
The evaluation results and models are logged by the corresponding WANDB project.


`Colab.ipynb` --reads--> `.yaml` --store in--> memory
`Colab.ipynb` --runs--> `Data` code --download--> data --stores in--> `Data` subfolder
`Colab.ipynb` --runs--> `Model` code --get--> Model class --initialize (passing configurations such as optimization, lr, etc.)--> Model
`Colab.ipynb` --runs--> `Augmentation` code --creates--> augmented data 
`Colab.ipynb` --calls train and provides data and augmented data to--> Model --creates--> trained model
`Colab.ipynb` --calls evalute and provides validate or test data--> Model --predicts--> label
`Colab.ipynb` --uploads--> results and parameters --to--> WANDB 


`Colab.ipynb` --uses--> function `Data_Load` --in--> `Data` code
`Colab.ipynb` --uses--> function `Aug` --in--> `Augmentation` code
`Colab.ipynb` --uses--> class `ModelClass` --in--> `Model` code



All images will be in [Batch size, Channel, Height, Width] format.




#`Data` Code
`Data_Load` Function Usage
Args:
	- task [str]: learning task of interest; default: "super" 
	- batch_size [int]: size of the data batch; default: "256" 
	- seed [int]: seed for randomization; default: "6759" 
Returns: 
	- labelledloader [torch.utils.data.DataLoader]: data loader for the training data with label
	- unlabelledloader [torch.utils.data.DataLoader]: data loader for the training data without label, for supervised learning, this is equal to `labelledloader`
	- validloader [torch.utils.data.DataLoader]: data loader for the validation data
	- testloader [torch.utils.data.DataLoader]: data loader for the test data   

	
#`Augmentation` Code    
`Aug` Function Usage   
Args:
	- data [Tensor]: input images with a size of [Batch size, Channel, Height, Width]
	- labels [Tensor]: Label class/target of the corresponding data
	- seed [int]: seed for randomization; default: "6759" 

Returns:
	- aug_data [Tensor]: output images with a size of [4*Batch size, Channel, Height, Width], where the first [Batch size, Channel, Height, Width] is the original data   
    - aug_label [Tensor]: Updated label class/target of the corresponding data. Equals to input `labels` for methods that do not change the label class/target


#`Model` Code   
Class `ModelClass` contians
def __init__
def forward - forward pass
def train_sup_up - train loop for supervised and unsupervised
def train_shot - train loop for few-shot	
def evaluation - Evaluate the models. Results stored into WANDB