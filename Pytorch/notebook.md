# Learn Pycharm 
## By Murillo Maiorano
```python
import torch, numpy
```
### Deep Learning Explained:
- In our day-by-day if we want to cook a chicken, we get the recipe and the ingredients, resulting in the cooked chicken.
In machine learning the idea is giving the model the cooked chicken and the ingredients, then find out the recipe.
Thats just one little example, but that idea spreads a long way.
- Those datas are represented in "tensors".

### Data:
- When we thing of machine learning, one of the first things that comes to mind is data.
- The data format used in machine learning are called "tensors" those are a blob of numbers, that represents something.
- Almost anything could be represented as a tensor.
- Even a image, for example, that its tensor would have rows and columns with the rgb code of each pixel.
- The idea of deep learning is to figure out padronization in the tensor and act accordingly.

## There are many ways to generate a tensor, being the most commonly used:

### 1. Example:
```python
tensor1 = torch.tensor(5)
tensor1
tensor(5)
``` 
    - Shape (tensor.shape): torch.Size([])
    - Dimensions (tensor.ndim): 0

### 2. Example:
```python
tensor2 = torch.tensor([5, 3])
tensor2
tensor([5, 3])
``` 
    - Shape (tensor.shape): torch.Size([2])
    - Dimensions (tensor.ndim): 1

### 3. Example:
```python
tensor3 = torch.rand(2, 5, 3)
tensor3
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
```
    - Shape (tensor.shape): torch.Size([2])
    - Dimensions (tensor.ndim): 1

Those tensors have different shapes and number of dimensions, 
ill be getting into that later, but here goes a fast explanation.
# Shape and Dimension:
## Shape:
   - Gotten by tensor.shape
   - Structure of the data.
   - For example, a shape of (1, 3, 4) indicates 1 block containing 3 rows, each with 4 columns.
   - Defined by its dimensions and the number of elements in each dimension.
   - The tensor1 in the example given earlier have 0 dimensions (Scalar), so it returns "torch.Size([])"
## Dimension:
   - Gotten by tensor.ndim
   - Amount of indices needed to access the value.

### Now that you already know what a tensor is, here is a simple way to understand its structure:
- tensor.ndim function shows the amount of the dimmensions the variable has
- tensor.shape function shows the shape of the variable

### Take for an example, the tensor:
```python
tensor([[[ 1,  2,  3, 5],
        [ 5,  6,  7, 3],
        [ 9, 10, 11, 2]]])
```
### Its properties shall be:
- Dimensions:
  - ```python
    tensor.ndim
    3
    ```
        - Explanation:
            - The dimensions are the account of the shape.
            - The number of dimensions corresponds to how many indices are needed to 
            access an element in the tensor.
- Shape:
  - ```python
    tensor.shape
    torch.Size([1, 3, 4])
    ```
        - Explanation:
          - Refers to the structure of the tensor.
          - 1 block with 3 rows each row having 4 columns (in that order)
            - Note:
              - Having a row with more or less columns may result in an error. 
              - Must be standardized.
## Other important functions:
### rand:
```python
rand = torch.rand(5) 
rand
tensor([0.7069, 0.6451, 0.1394, 0.0382, 0.9321])
```
    - Returns a tensor with random numbers (0 to 1) based on the shape given on the input.

### zeros:
```python
zeros = torch.zeros(1, 3, 4)
zeros
tensor([[[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]])
```
    - Returns a tensor with "0" based on the shape given on the input.

### ones:
```python
ones = torch.ones(1, 3, 4)
ones
tensor([[[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]])
```
    - Returns a tensor with "1" based on the shape given on the input.

### arange:
```python
arange = torch.arange(0, 10)
arange
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arange2 = torch.arange(0, 10, 2)
arange2
tensor([0, 2, 4, 6, 8])
```
    - Returns a tensor with a range of numbers based on the given input.
    - The first input defines the start, the second input defines the end and the third defines if it has steps.

## Concepts worth noting:

### Scalar:
- 1 by 1 data (0 dimensions)
  - ```python
    scalar = torch.tensor(5)
    scalar
    tensor(5)
    ```
### Vector:
- 1 by n data (1 dimension)
  - ```python
    vector = torch.tensor([1, 2])
    vector
    tensor([1, 2])
    ```
### Matrix:
- n by m data (2 dimensions)
  - ```python
    MATRIX = torch.tensor([[1, 2],
        [2, 3]])
    MATRIX
    tensor([[1, 2],
        [2, 3]])
    ```
### Tensor:
- n by m by p data (3 dimensions)
  - ```python
    TENSOR = torch.tensor([[
                [1, 2, 3, 4],
                [5, 6, 7, 4],
                [9, 10, 11, 5]
            ]])
    TENSOR
    tensor([[[ 1,  2,  3,  4],
            [ 5,  6,  7,  4],
            [ 9, 10, 11,  5]]])
    ```
### Nodes:
Now that we understand what is a tensor, its structure and how it is treated, we shall begin to understand how to process
that data.
That is thru nodes.
Nodes are those dots in those famous neural network images, each dot its a node and has a corresponding meaning.
Nodes alone are useless, they have an use when rearanged to fit in a model.

Some common inputs for the neural network are: variables with weights and a bias.
The weight of the variable is defined as how important it is (on the math).
Bias is a flat number accounted.

### Math example:
The formula y = (x * w) + b
Where:
  - y = Result
  - x = Input data
  - w = Weight of the input
  - b = Bias
  - y = (3⋅* 2) + 1 = 7

### Model example:
  ```python
class SimpleModel(nn.Model):
  def __init__(self): #Define the shape of the model and each node function.
      super(SimpleModel, self).__init__()
      self.linear1 = nn.Linear(1, 3) #1 Input data and 3 hidden nodes
      self.relu = nn.ReLU() #Makes so those 3 nodes are ReLU activate(If number > 0 return number, else return 0)
      self.linear2 = nn.Linear(3, 1) #After the 3 hidden nodes, 1 output node
  def forward(self, x): #Forward the input
      x = self.linear1(x)
      x = self.relu(x)
      x = self.linear2(x)
      return x
model = SimpleModel()
input_data = torch.tensor([[2.0]])
output = model(input_data)
print(f"Output: {output.item()}")
  ```
But how is it learning? You might ask. Its a deep without learn.
Well, that is all in the fact that we didnt measure the error.
The way the model learns is by measuring the error and making minor changes to the numbers in the nodes, in order to get
a more accurate result.
For that, many famous libraries exists just to measure the error.

A famous library is called **MSE** (Mean Squared Error).
Suppose we run a neural network on an apartment with $1000/mo rent, but the network predicts $500/mo. The simplest way 
to measure the loss in this case is to calculate the difference:
500 − 1000 = −500
The difference indicates that the prediction was 500 dollars below the actual rent.

For just one data point, the difference seems reasonable. But imagine we have a second apartment with rent $1500, but the
model overestimates the rent as $2000. The difference in this case is:
2000 − 1500 = 500
With one loss of 500 and another of -500, the average loss for the model is actually 0! But that doesn’t make sense, 
since this network isn’t perfectly accurate.

MSE makes differences positive by squaring them. To calculate MSE on our two example apartments, we would:
 - Calculate the differences: 500 and -500
 - Square both: 500^2 and (-500)^2
 - Take the average:
   - ((500 + 1000)^2 + (1500 - 1000)^2) / 2 = 250,000
 - A loss of 250,000 seems to be too high, but remember that we squared that number. 
 - So the square root of 250,000 = 500 is our actual loss.

Great on the theory!
### How do we actually do it?
````python
loss = nn.MSELoss()
````
Now that we initialized the MSE, we can calculate the loss by sending 2 inputs.
- input_data1 = The predicted values.
- input_data2 = The actual target values.
```python
loss = nn.MSELoss()
predictions = torch.tensor([500,2000],dtype=torch.float)
target = torch.tensor([1000,1500],dtype=torch.float)
print(loss(predictions,target))
```