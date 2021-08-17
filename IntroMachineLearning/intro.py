# -*- coding: utf-8 -*-

#polinomio de grado 3 que se aproxime al seno de pi
# es en un sistema fundamental lo que hace un red neuronal
#porque esta tratando de aproximarse a un resultado


import torch
import math
import matplotlib as plt

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
#tienes una funcion y 2000 puntos que se pueden aproximar
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)

y = torch.sin(x)

#Tensors : a, b, c, d
#numbers put together

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

#Y= a + bx + cx^2 + dx3
#se inicializa cada valor del polinomios en valores rand


learning_rate = 1e-6
for t in range(2000):#for loop para pasar por todos los puntos
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
#funcion que maneja el programa

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    #loss representa la diferencua entre el valor que estamos obteniendo y el valor que debe ser
    if t % 100 == 99:
        print(t, loss.item())
    #si el numero es cercano lo imprime

    #calculamos el loss para que nosotros sepamos como arreglar ese problrema
    #

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()
    #esta funcion es la forma en la que la maquinna esta viendo como estan saliendo los resultado
    #y como deberian ajustarlos

    #gradient es una forma abstracta de decirnos la pendiente, que tanto nos alejamos o acercamos

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    
    #aqui se estan usando esas slopes (pendientes) para ajustar el resultado
    with torch.no_grad():
        a -= learning_rate * a.grad #al valor de a se le resta el learning rate que se multiplica por el slope
        b -= learning_rate * b.grad #a.grad nos dice la pendiente del punto respecto a la parabola
        c -= learning_rate * c.grad # usamos en menos porque nos podemos mover a ambos lados de la parabola, izq o derecha
        d -= learning_rate * d.grad #el learning rate es que tanto queremos que cambie el valor de las variables abcd

        # Manually zero the gradients after updating weights
        a.grad = None #reset the gradients para la siguiente iteracion
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')