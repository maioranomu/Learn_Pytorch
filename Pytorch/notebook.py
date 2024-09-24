import numpy
import torch

"""
The data format used in machine learning is called "tensors" those are a blob of numbers, that represents something.
Almost anything could be represented as a tensor.
Even a image, for example, that its tensor would have rows and columns with the rgb color of each pixel.
Thats just one little example, but that idea spreads a long way.

There are many ways to generate a tensor, being the most commonly used:
    tensor1 = torch.tensor(5)
        returns:
            tensor(5)
            {
                Shape (tensor.shape): torch.Size([])
                Dimensions (tensor.ndim): 0
            }
    tensor2 = torch.tensor([5, 3])
        returns:
            tensor([5, 3])
            {
                Shape (tensor.shape): torch.Size([2])
                Dimensions (tensor.ndim): 1
            }
    tensor3 = torch.rand(2, 5, 3)
        returns:
            tensor([[[0.2184, 0.9275, 0.6348],
                [0.2078, 0.5599, 0.6421],
                [0.2963, 0.1520, 0.4446],
                [0.1961, 0.9393, 0.1961],
                [0.2563, 0.7623, 0.6705]],

                [[0.1698, 0.9152, 0.6661],
                [0.0963, 0.9766, 0.3432],
                [0.7947, 0.0277, 0.2804],
                [0.5053, 0.6368, 0.9621],
                [0.2969, 0.1018, 0.9458]]])
            {
                Shape (tensor.shape): torch.Size([2, 5, 3])
                Dimensions (tensor.ndim): 3
            }
    Those tensors have different shapes and number of dimensions, ill be getting into that later.
    But here is a simple explanation:
        Shape:
            Structure of the data.
            For example, a shape of (1, 3, 4) indicates 1 block containing 3 rows, each with 4 columns.
            Defined by its dimensions and the number of elements in each dimension.
            The tensor1 in the example given have 0 dimensions (Scalar), so it returns "torch.Size([])"
        Dimensions:
            Amount of indices needed to access the value.

Now that you already know what a tensor is, here is a simple way to understand its structure:
    tensor.ndim function shows the amount of the dimmensions the variable has
    tensor.shape function shows the shape of the variable

    Those really help to understand the way a tensor works.
    
Take for an example, the tensor:
    tensor([[[ 1,  2,  3, 5],
            [ 5,  6,  7, 3],
            [ 9, 10, 11, 2]]])
         
Its settings shall be:
    - DIMENSIONS:
        3
        Explanation:
            The dimensions are the account of the shape.
            The number of dimensions corresponds to how many indices are needed to access an element in the tensor.
        
    - SHAPE:
        torch.Size([1, 3, 4])
        Explanation:
            Refers to the structure of the tensor.
            1 block with 3 rows each row having 4 columns (in that order)
            Note:
                Having a row with more or less columns may result in an error. Must be standardized.

Other important functions are:
    rand = torch.rand(5) 
        returns:
            tensor([0.7069, 0.6451, 0.1394, 0.0382, 0.9321])
            Explanation:
                Returns a tensor with random numbers (amount of nums are the input) from 0 to 1 and returns them in a list of tensors

    zeros = torch.zeros(1, 3, 4)
        returns:
            tensor([[[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]])
            Explanation:
                Returns a tensor with "0" based on the shape given on the input.
                
    ones = torch.ones(1, 3, 4)
        returns:
            tensor([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
            Explanation:
                    Returns a tensor with "1" based on the shape given on the input.
    
    arange = torch.arange(0, 10)
        returns:
            tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            Explanation:
                Returns a tensor with a range of numbers based on the given input.
    arange2 = torch.arange(0, 10, 2)
        returns:
            tensor([0, 2, 4, 6, 8])
            Explanation:
                The first input defines the start, the second input defines the end and the third defines if it has steps.
                
    arange = torch.arange(0, 10)
    like = torch.zeros_like(arange)
        returns:
            tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            Explanation:
                Creates a tensor similar to the tensor in the input.
                Creates a tensor with same dimensions and shape, but different values.
                
Concepts worth remembering:
    Scalar: 1 by 1 data (0 dimensions)
        Example:
            scalar = torch.tensor(5)
            returns:
                tensor(5)

    Vector: 1 by n data (1 dimension)
        Example:
            vector = torch.tensor([1, 2])
            returns:
                tensor([1, 2])
        
    Matrix: n by m data (2 dimensions)
        Example:
        matrix = torch.tensor([[1, 2],
                [2, 3]])
            returns:
                tensor([[1, 2],
                    [2, 3]])
            
    Tensor: n by m by p data (3 dimensions)
        Example:
            TENSOR = torch.tensor([[
                [1, 2, 3, 4],
                [5, 6, 7, 4],
                [9, 10, 11, 5]
            ]])
                returns:
                    tensor([[[ 1,  2,  3,  4],
                        [ 5,  6,  7,  4],
                        [ 9, 10, 11,  5]]])
                    
"""