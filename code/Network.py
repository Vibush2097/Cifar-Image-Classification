### YOUR CODE HERE
import torch
import torch.nn as nn
from torch.functional import Tensor

"""This script defines the network.
"""

class MyNetwork(nn.Module):

    def __init__(self,
                resnet_version,
                resnet_size,
                num_classes,
                first_num_filters,
        ):
        
        super(MyNetwork, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        self.start_layer = nn.Conv2d(3, self.first_num_filters, kernel_size=3, stride=1, padding=1, bias=False)

        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
            )
        block_fn = []
        block_fn = standard_block
            
        self.stack_layers = nn.ModuleList()
        
        for i in range(4):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters, self.resnet_version, self.num_classes, False)
        
        
    def __call__(self, inputs, training):
        '''
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
        '''
        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        outputs = self.start_layer(inputs)
        outputs = self.batch_norm_relu_start(outputs)            
            
        for i in range(4):
            outputs = self.stack_layers[i](outputs)
            
        outputs = self.output_layer(outputs, training)
        
        return outputs
    
class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features) -> None:
        super(batch_norm_relu_layer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, device = torch.device("cuda:0"))
        self.relu = nn.ReLU()
        
    def forward(self, inputs: Tensor) -> Tensor:
        output = self.bn(inputs)
        output = self.relu(output)
        return output
    
class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
                 convolution.
        projection_shortcut: The function to use for projection shortcuts
                             (typically a 1x1 convolution when downsampling the input).
        strides: A positive integer. The stride to use for the block. If
                 greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
                           first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.filters, self.stride, self.projection_shortcut = filters, strides, projection_shortcut
        
        self.conv1 = nn.Conv2d(filters//strides, filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn_relu = batch_norm_relu_layer(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        residual = inputs
        
        output = self.conv1(inputs)
        output = self.bn_relu(output)
        
        output = self.conv2(output)
        output = self.bn(output)
        
        if self.projection_shortcut is not None:
                residual = self.projection_shortcut(inputs)
                
        output = output + residual
        output = self.relu(output)
        # output = self.dropout(output)
        return output

        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
                 convolution in a block.
        block_fn: 'standard_block' or 'bottleneck_block'.
        strides: A positive integer. The stride to use for the first block. If greater than 1, this layer will 
                 ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer 
        first_num_filters: An integer. The number of filters to use for the first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters
        self.filters = filters_out
        self.stride = strides
        self.first_num_filters = first_num_filters
            
        blocks = []
    
        if self.stride != 1:
            self.projection_shortcut = nn.Sequential(
                nn.Conv2d(self.filters//self.stride, self.filters, kernel_size=1, stride=self.stride, padding=0, bias=False),
                nn.BatchNorm2d(self.filters),
            )
        
        if(self.filters == self.first_num_filters):
                for i in range(resnet_size):
                    blocks.append(standard_block(self.filters, None, self.stride, self.first_num_filters))
                self.blocks = nn.Sequential(*blocks)
        else:
            for i in range(resnet_size):
                if i==0:
                    blocks.append(standard_block(self.filters, self.projection_shortcut, self.stride, self.first_num_filters))
                else:
                    blocks.append(standard_block(self.filters, None, 1, self.first_num_filters))
            self.blocks = nn.Sequential(*blocks)
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(inputs)

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes, training) -> None:
        super(output_layer, self).__init__()

        self.bn_relu = batch_norm_relu_layer(filters)
        self.filters, self.num_classes = filters, num_classes
        
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(filters, num_classes)
#         self.dropout = nn.Dropout(p=0.5)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor, training) -> Tensor:
        output = self.avgpool(inputs)  # 1x1
        output = output.view(output.size(0), -1)
#         output = self.dropout(output)
        output = self.fc(output)
        return output


### END CODE HERE