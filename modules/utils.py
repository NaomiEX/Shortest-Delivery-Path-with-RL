import numpy as np

def softmax(x):
    """Performs softmax on input x

    Args:
        x (Array or np.array): input array

    Returns:
        Array or np.array: softmax results between 0-1
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_min_steps(v_a, v_b):
    """
    Calculate the minimum Manhatten distance between 2 vector points
    
    Args:
        v_a: First vector point
        v_b: Second vector point
    
    Return:
        float: manhatten distance between 2 vector point
    """
    assert isinstance(v_a, list) and isinstance(v_b, list) and len(v_a) == len(v_b) == 2, \
        f"Expected list inputs of size 2 but instead got v_a: {v_a}, v_b: {v_b}"
    return abs(v_b[0] - v_a[0]) + abs(v_b[1] - v_a[1])
