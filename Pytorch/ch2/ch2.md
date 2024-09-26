# Learn Pycharm Ep. 2
## By Murillo Maiorano
 - 2024, 16 years old.
 - On this document, I will share all my journey to learn pytorch and machine learning.
 - Im mostly doing that in a matter to learn faster while having fun, but also having hopes on that in the future, look back to that and see how much I evolved.
 - Dont be harsh on English errors, thats my second language.
    - If you find any, please reach out so I can fix it.
- Keep in mind thats a for fun project and I dont recommend using it to study, since im studying myself.
#
```python
import torch
```
In the last chapter we covered the basics of tensor structure and model structure, this one will start with basic operations that can be done to tensors.

# Add, Subtract, Multiply, Divide.
```python
tensor1 = torch.rand(2, 3)
print(tensor1)
tensor([[0.1867, 0.4248, 0.0543],
        [0.3087, 0.8895, 0.5518]])

tensor2 = torch.rand(2, 3)
print(tensor2)
tensor([[0.7015, 0.7288, 0.0089],
        [0.1472, 0.1562, 0.7006]])

tensor3 = tensor1 + tensor2
print(tensor3)
tensor([[-0.5149, -0.3040,  0.0454],
        [ 0.1615,  0.7333, -0.1487]])

tensor3 = tensor1 - tensor2
print(tensor3)
tensor([[-0.5149, -0.3040,  0.0454],
        [ 0.1615,  0.7333, -0.1487]])

tensor3 = tensor1 * tensor2
print(tensor3)
tensor([[0.1309, 0.3096, 0.0005],
        [0.0454, 0.1390, 0.3866]])

tensor3 = tensor1 / tensor2
print(tensor3)
tensor([[0.2661, 0.5829, 6.0960],
        [2.0968, 5.6942, 0.7877]])
```
 Overall basic math operations, easy to understand, just make sure to pay attention to the way the math occurs.

 Notice that by the way that the count is done, if you try to do any operation with tensors of different shapes, an error may occur.
### References:
- FreeCodeCamp Video:
    - https://youtu.be/V_xro1bcAuA?si=B4sRN7YLOzgzy--c
- CodeCademy Intro to Pytorch and Neural Networks Course
    - https://www.codecademy.com/enrolled/courses/intro-to-py-torch-and-neural-networks