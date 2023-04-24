import numpy as np
from Value import Value
from typing import List


class Neuron:
    """
    Multiple inputs, only one output.
    """

    def __init__(self, nin):
        self.w = [Value(data=np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(data=np.random.uniform(-1, 1))

    def __call__(self, x):
        """This is forward pass"""
        activation = sum([_w * _x for _w, _x in zip(self.w, x)]
                         )  # this is activation signal
        activation += self.b
        o = activation.tanh()
        return o

    def parameters(self):
        return [self.b, *self.w]


class Layer:
    """
    Multiple inputs, multiple outputs.
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) > 1 else outs[0]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
        # params = []
        # for n in self.neurons:
        #     params.extend(n.get_parameters)
        # return params


class MLP:
    """
    Multi-layer perceptor module. Multiple layers (each with multiple inputs/outputs), and one final output.
    """

    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin, *nouts]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)  # output of a layer is input to the next layer
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
