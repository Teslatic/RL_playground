import numpy as np

def convert_vector2tensor(vector):
    shape = (1, vector.shape[0])
    return vector.reshape(shape)

def convert_tensor2vector(tensor):
    return tensor.reshape(-1)
