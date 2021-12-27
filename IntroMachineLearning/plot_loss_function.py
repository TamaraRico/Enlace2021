# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Same code as before but we are using graphs to see the changes and changing y function to experiment with new aproximations 
Reference code: https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_autograd.html#sphx-glr-beginner-examples-autograd-polynomial-autograd-py

"""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda")  # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)

y = torch.exp(x)
# y = torch.sin(x)

ar=np.array([]) # Array holding the loss values for future graph
numbers=np.arange(2000)

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)
e = torch.randn((), device=device, dtype=dtype, requires_grad=True)
#d = torch.tensor(1e-20, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-9
loss_old = None
for t in range(20000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4
    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99: 
        print(t, loss.item())
    ar=np.append(ar, loss.item())
        
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        e -= learning_rate * e.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
        e.grad = None

    # ajustar el learning rate para que el resultado sea mas exacto
    if loss_old is not None:
         if loss.item() < loss_old:
             learning_rate = learning_rate * 2
         else:
             learning_rate = learning_rate / 2
    loss_old = loss.item()

    # Plot each iteration
    a1 = a.detach().cpu().numpy()
    b1 = b.detach().cpu().numpy()
    c1 = c.detach().cpu().numpy()
    d1 = d.detach().cpu().numpy()
    e1 = e.detach().cpu().numpy()
    x1 = x.detach().cpu().numpy()
    y1 = y.detach().cpu().numpy()

    # plt.ion()
    # plt.clf()
    # plt.plot(x1,y1, label="real")
    # plt.plot(x1, a1 + b1*x1 + c1*x1**2 + d1*x1**3 + e1*x1**4, label="approx")
    # plt.legend()
    # plt.xlim([-math.pi,math.pi])
    # plt.ylim([-2,2])
    # plt.pause(1e-10)
    
    # plt.figure(2)
    # plt.plot(numbers,ar, label="loss")
    # plt.xlim([-100,100])
    # plt.ylim([-100,100])
    # plt.show()



print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

plot1=plt.figure(1)
# Plot each iteration
a1 = a.detach().cpu().numpy()
b1 = b.detach().cpu().numpy()
c1 = c.detach().cpu().numpy()
d1 = d.detach().cpu().numpy()
e1 = e.detach().cpu().numpy()
x1 = x.detach().cpu().numpy()
y1 = y.detach().cpu().numpy()

plt.plot(x1,y1, label="real")
#plt.plot(ar, label="loss")
plt.plot(x1, a1 + b1*x1 + c1*x1**3 + d1*x1**3 + e1*x1**4, label="approx")
plt.legend()
plt.title("Approching a function using pytorch")
plt.xlim([-math.pi, math.pi])
plt.ylim([-2,2])

plot2=plt.figure(2)
plt.semilogy(ar)
plt.title("Loss")
#plt.plot(ar, label="loss")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss") # Mean Squared Errors
plt.show()

