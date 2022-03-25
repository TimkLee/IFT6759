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
The `Model` folder contains code for generating the desired model structure. The performance parameters are saved in the corresponding text file. (Default: ALL-CNN)
The `Evaluation` folder contains code for evaluating the model performance. The evaluation parameters/images are saved in the corresponding subfolder.


`main.py` --reads--> `.yaml` --store in--> memory
`main.py` --runs--> `Data` code --download--> data --stores in--> `Data` subfolder
`main.py` --runs--> `Model` code --get--> Model class --initialize (passing configurations such as optimization, lr, etc.)--> Model
`main.py` --runs--> `Augmentation` code --creates--> augmented data --store in--> `Augmentation` folder
`main.py` --calls train and provides data and augmented data to--> Model --creates--> trained model
`main.py` --calls evalute and provides validate or test data--> Model --predicts--> label
`main.py` --runs--> `Evaluation` code --reads--> performance parameter --produces--> evaluation parameters/images --store in--> `Evaluation` folder 


`main.py` --uses--> function `Data_Load` --in--> `Data` code
`main.py` --uses--> function `Aug` --in--> `Augmentation` code
`main.py` --uses--> class `ModelClass` --in--> `Model` code
`main.py` --uses--> function `Eval` --in--> `Evaluation` code


All images will be in [Batch size, Channel, Height, Width] format.


To run `main.py, e.g. python main.py -c Example.yaml -d cpu


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
	- seed [int]: seed for randomization; default: "6759" 

Returns:
	- aug_data [Tensor]: output images with a size of [4*Batch size, Channel, Height, Width], where the first [Batch size, Channel, Height, Width] is the original data   
   

#`Model` Code   
Class `ModelClass` contians
def __init__
def forward - forward pass
def train - train loop for supervised and unsupervised
def shottrain - train loop for few-shot	
def evaluating - for validation and testing
	
	
	
	
	
	
	