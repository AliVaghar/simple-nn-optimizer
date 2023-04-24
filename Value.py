import numpy as np


class Value:

    def __init__(self, data, children=(), _op='', label='') -> None:
        """
        data (float): the scalar value of the node.
        children (tuple): a tuple containing all children we used in finding data.
        """
        self.data = data
        self.grad = 0
        self._prev = set(children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data,
                    children=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __radd__(self, other):  # other * self
        return self + other

    def __truediv__(self, other):  # self / other
        return self * (other ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int or float"
        out = Value(self.data ** other, children=(self, ),
                    _op=f"**{other}", label=f"{self.label} ** {other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def tanh(self):
        out = Value(data=np.tanh(self.data), children=(self, ), _op='tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(data=np.exp(self.data), children=(self, ), _op='exp')

        def _backward():
            self.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def topological_sort(g):
            if g not in visited:
                visited.add(g)
                for c in g._prev:
                    topological_sort(c)
                topo.append(g)
        topological_sort(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
