import numpy as np
from tensorflow import convert_to_tensor

def convert_scalar2vector(scalar):
    return [scalar]

def convert_scalar2tensor(scalar):
    # vector = convert_scalar2vector(scalar)
    # print(vector)
    return [[scalar]]

def convert_vector2tensor(vector):
    # print("Vektor: {}".format(vector))
    shape = (1, vector.shape[0])
    # print("Vektor: {}".format(shape))
    return vector.reshape(shape)

def convert_tensor2vector(tensor):
    return tensor.reshape(-1)
