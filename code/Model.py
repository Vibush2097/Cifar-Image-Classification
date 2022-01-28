### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from Network import MyNetwork
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.config = configs        
        self.network = MyNetwork(
            self.config.model_version,
            self.config.depth,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        
        self.test_loss = []
        self.best_acc = 0
        
    def model_setup(self):
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, momentum=self.config.momentum)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // configs['batch_size']
        max_epoch = self.config.max_epochs
        save_interval = configs['save_interval']

        print('### Training... ###')
        
        accuracy_values = []
        train_loss_values = []
        iterations = []
        
        for epoch in range(1, max_epoch+1):
            self.network.train()
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            batch_size = configs['batch_size']
            
            if epoch >= 30 and epoch%10==0:
                self.config.learning_rate = self.config.learning_rate * (1/(1+self.config.lr_decay*epoch))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate
                        
            sum_correct = 0
            total = 0
            accuracy = 0
            loss = 0
            train_loss = 0
            
            for i in range(num_batches):
                curr_x_train_batch = curr_x_train[i*batch_size: min((i+1)*batch_size, num_samples)]
                curr_y_train_batch = curr_y_train[i*batch_size: min((i+1)*batch_size, num_samples)]
                                
                curr_x_train_batch = torch.tensor([parse_record(x, True) for x in curr_x_train_batch]).float()
                y_tensor = torch.tensor(curr_y_train_batch).long()
                if torch.cuda.is_available():
                    y_tensor = y_tensor.cuda()
                    curr_x_train_batch = curr_x_train_batch.cuda()
                
                self.optimizer.zero_grad()
                outputs = self.network(curr_x_train_batch, True)
                loss = self.loss(outputs, y_tensor)
                loss.backward()
                self.optimizer.step()
                
                preds = torch.argmax(outputs, dim=1)
                y_actual = y_tensor
                                
                sum_correct += torch.sum(preds==y_actual)
                total += batch_size
                accuracy = sum_correct/total
                train_loss += loss.item()*batch_size      

                print('Batch {:d}/{:d} Loss {:.6f} Accuracy {:.6f}'.format(i, num_batches, loss, accuracy), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Train Loss {:.6f} Accuracy {:.6f} Duration {:.3f} seconds.'.format(epoch, (train_loss/num_samples), accuracy, duration))
            
            if x_valid is not None and y_valid is not None:
                self.evaluate(x_valid, y_valid)
                # print("score = {:.6f} in validation set.\n".format(score))
            
            if epoch % save_interval == 0:
                self.save(epoch)
                
            accuracy_values.append(accuracy.item())
            train_loss_values.append(train_loss/num_samples)
            iterations.append(epoch)
            
            # self.scheduler.step()
        
        self.train_loss = train_loss_values
        
        plt.plot(train_loss_values, label='Training Loss')
        plt.plot(self.test_loss, label='Testing Loss')
        plt.title("Train vs Test Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Testing Loss")
        plt.legend(frameon=True)
        plt.show()
        
        plt.plot(iterations, accuracy_values)
        plt.title("Training Accuracy Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Training Accuracy")
        plt.show()
            

    def evaluate(self, x, y):
        x_evaluate = [parse_record(ip, False) for ip in x]
        Y_tensor = torch.tensor(y).long()
        num_samples = x.shape[0]
        num_batches = 1
        
        if torch.cuda.is_available():
            self.network = self.network.cuda()
            Y_tensor = Y_tensor.cuda()
        
        self.network.eval()
        preds = []
        # test_loss = 0
            
        with torch.no_grad():
            batch_size = 100 if num_samples > 100 else num_samples
            num_batches = num_samples//batch_size
            for i in range(num_batches):
                x_cur_batch = x_evaluate[i*batch_size : min((i+1)*batch_size, num_samples)]
                # y_cur_batch = y[i*batch_size: min((i+1)*batch_size, num_samples)]
                
                X_tensor = torch.tensor(x_cur_batch).float()
                # y_tensor = torch.tensor(y_cur_batch).long() 
                
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                    # y_tensor = y_tensor.cuda()
                    
                outputs = self.network(X_tensor, False)
                # loss = self.loss(outputs, y_tensor)
                # test_loss += loss.item()*batch_size
                
                preds.append(outputs)
                
        preds = torch.cat(preds, dim=0)
        test_loss = self.loss(preds, Y_tensor)
        # print("Test loss: ", (test_loss/num_samples))
        print("Test loss: ", test_loss.item())
        # self.test_loss.append((test_loss/num_samples))
        self.test_loss.append(test_loss.item())
        preds = torch.argmax(preds, dim=1)
        score = torch.sum(preds==Y_tensor)/y.shape[0]
        
        if score > self.best_acc:
            print("Previous best: ", self.best_acc)
            print("Current best: ", score.item())
            self.best_scc = score.item()
            self.save_best()
        
        print("score = {:.6f} in validation set.\n".format(score))
        # return score

    def predict_prob(self, x):
        x = [ip.reshape(32,32,3) for ip in x]
        x = [np.transpose(ip, (2, 0, 1)) for ip in x]
        x = np.array(x)
        print(x.shape)
        X_tensor = torch.tensor([parse_record(rec, False) for rec in x]).float()
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            self.network = self.network.cuda()
        
        self.network.eval()
        preds = []
        with torch.no_grad():
            preds = self.network(X_tensor, False)
        softmax = nn.Softmax(dim=1)
        preds = softmax(preds)
        return preds
    
    def test_or_validate(self, x, y, checkpoint_num_list, test_or_validate):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.save_dir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            batch_size = 100 if 100 < x.shape[0] else x.shape[0]
            for i in tqdm(range(x.shape[0]//batch_size)):
                ### YOUR CODE HERE
                cur_batch = x[i*batch_size: min((i+1)*batch_size, x.shape[0])]
                cur_batch = torch.tensor([parse_record(x, False) for x in cur_batch]).float()
                
                if torch.cuda.is_available():
                    cur_batch = cur_batch.cuda()
                    y = torch.tensor(y).cuda()
                    
                outputs = self.network(cur_batch, False)
                preds.append(outputs)

            preds = torch.cat(preds, dim=0)
            final_prediction = torch.argmax(preds, dim=1)

            if test_or_validate == "test":
                print('Test accuracy: {:.4f}'.format(torch.sum(final_prediction==y)/y.shape[0]))
            else:
                print('Validation accuracy: {:.4f}'.format(torch.sum(final_prediction==y)/y.shape[0]))

    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.save_dir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.save_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.\n")
        
    def save_best(self):
        checkpoint_path = os.path.join('best_model', 'best_model_2.ckpt')
        os.makedirs('best_model', exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Saved the best model.\n")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cuda:0")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE