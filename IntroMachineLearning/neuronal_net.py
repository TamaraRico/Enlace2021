# -*- coding: utf-8 -*-
"""
PyTorch Approximating a function using PyTorch Neural Network Module
A third order polynomial, trained to predict y=sin(x) from -pi to pi
by minimizing squared Euclidean distance.
https://pytorch.org/tutorials/beginner/examples_nn/polynomial_nn.html#sphx-glr-beginner-examples-nn-polynomial-nn-py
"""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# Tensors: numbers put together
# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3]) # Exponentes
xx = x.unsqueeze(-1).pow(p) # Elevar x a cada potencia, esta creando el polinomio mediante vectores 

loss_array=np.array([])
# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential( 
    torch.nn.Linear(3, 1), # here numpy is making the polynomial function
    # this allow us to have multiple layers for super resolution networks 
    torch.nn.Flatten(0, 1)
)

# nn.Linear (3 entradas, 1 salida)
# las entradas son los numeros random de los coeficientes y se crea como funcion

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')
# es la misma loss, (y-y_pred)^2+()^2+()^2+()^2+()^2

learning_rate = 1e-6
for t in range(2000):    
    # you usually have training data and testing data
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(xx)
    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the loss.
    loss = loss_fn(y_pred, y) # same loss using the functioon loss_fn
    if t % 100 == 99:
        print(t, loss.item())
    # numpy vector for the loss values 
    loss_array=np.append(loss_array, loss.item()) # a partir de este vector podemos graficar la pérdida
    
    # Zero the gradients before running the backward pass.
    model.zero_grad()
    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            
    # Pot the functions to see what's going on, for me 
    # x1 = x.detach().cpu().numpy()
    # y1 = y.detach().cpu().numpy()
    # xx1 = xx.detach().cpu().numpy() my attemp
    # y_p1=y_pred.detach().cpu().numpy() my attemp 
    # apx=model(xx).detach().cpu().numpy()

    # plt.ion()
    # plt.clf()
    # plt.plot(x1,y1, label="real")
    # plt.plot(x1, y_p1, label="approx") not sure 
    # plt.plot(np.linspace(-math.pi, math.pi, 2000), apx, label="approx")
    # plt.legend()
    # plt.xlim([-math.pi,math.pi])
    # plt.ylim([-2,2])
    # plt.pause(1e-10)

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
print(f"Final MSE Loss: {loss_array[-1]}")

# Create plots for evaluation
x1 = x.detach().cpu().numpy()
y1 = y.detach().cpu().numpy()
approx = model(xx).detach().cpu().numpy()

plt.figure()
plt.plot(x1,y1, label="real")
plt.plot(x1, approx, label="approx")
plt.legend()
#plt.xlim([-math.pi, math.pi])
#plt.ylim([-2,2])

plt.figure()
plt.semilogy(loss_array)
plt.title("Loss during training")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.show()

# Save data for later comparision
# np.save("axis.npy", x1)
# np.save("true.npy", y1)
# np.save("nn_approx.npy", approx)
