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
`main.py` --runs--> `Augmentation` code --creates--> augmented data --store in--> `Augmentation` folder
`main.py` --runs--> `Model` code --trains with--> augmented data --produces--> performance parameters --store in--> `Model` folder  
`main.py` --runs--> `Evaluation` code --reads--> performance parameter --produces--> evaluation parameters/images --store in--> `Evaluation` folder 

`main.py` --uses--> function `Data_Load` --in--> `Data` code
`main.py` --uses--> function `Aug` --in--> `Augmentation` code
`main.py` --uses--> function `Train` --in--> `Model` code
`main.py` --uses--> function `Eval` --in--> `Evaluation` code



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
	
    
   
   
   
   

	
	
	
	
	
	