# Cifar-Image-Classification
This was a project done as a part of the CSCE636 Deep Learning course at Texas A&amp;M University  
  
The document serves as a guide on how to run the deep learning project.  

For training:  
usage: main.py [-h] [--save_dir SAVE_DIR] [--depth DEPTH] [--num_classes NUM_CLASSES]  
               [--model_version MODEL_VERSION] [--first_num_filters FIRST_NUM_FILTERS] [--weight_decay WEIGHT_DECAY]  
               [--learning_rate LEARNING_RATE] [--lr_decay LR_DECAY] [--momentum MOMENTUM] [--max_epochs MAX_EPOCHS]  
               mode  
  
To begin training, you can simply run  
python main.py train  
  
After training, the best model is saved to /best_model/best_model.ckpt  
This saved model is picked up by default while testing and doing predictions on the private data set  
  
For Testing:  
python main.py test  
  
For prediction:  
python main.py predict  
  
The predictions are stored in a folder titled 'private_data' under the name predictions.npy  
  
During training, the default values for the hyperparameters are as follows:  
SAV_DIR = /saved_models/  
DEPTH = 2 // This is the number of blocks in each layer  
NUM_CLASSES = 10  
MODEL_VERSION = 1 //This value is always 1  
FIRST_NUM_FILTERS = 32  
WEIGHT_DECAY = 1e-4  
LEARNING_RATE = 0.01  
LR_DECAY = 1e-4  
MOMENTUM = 0.9  
MAX_EPOCHS = 200  
  
NOTE:  
The project code does not include the cifar-10 batches. In order to load the cifar 10 data, kindly store the 5 training batches and test batch in a folder titled 'cifar-10-batches-py'.  
cifar-10-batches-py/data_batch_1  
cifar-10-batches-py/data_batch_2  
cifar-10-batches-py/data_batch_3  
cifar-10-batches-py/data_batch_4  
cifar-10-batches-py/data_batch_5  
cifar-10-batches-py/test_batch  
