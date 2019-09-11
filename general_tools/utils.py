import numpy as np


def distance_from_the_line(line_start, line_end, point):
    """
    Computes the distance of a point from a given line
    http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    :param line_start: (np.array) - first point of the line 
    :param line_end: (np.array) - second point of the line 
    :param point: (np.array)
    :return: (float) point distance from a given line
    """
    return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) /\
           np.linalg.norm(line_start - line_end)


def map_to_range(value, src_range=(-1.0, 1.0), dst_range=(0.0, 1.0)):
    """
    Shifts the passed value from one range to another
    
    :param value: (float) - value to shift
    :param src_range: (tuple) - source range
    :param dst_range: (tuple) - destination range
    :return: (float) shifted value
    """
    a, b = src_range
    c, d = dst_range

    return float(c + ((d - c) / (b - a)) * (value - a))


def create_symmetric_range(n):
    """
    Creates symmetric zero centered array with n elements between [-1.0, 1.0]  

    :param n: (int) - number of elements of the resulting array
    :return: (numpy.array)
    """
    assert n >= 3 and n % 2 != 0
    
    temp = np.linspace(-1.0, 1.0, n - 1, True).tolist()
    temp.insert(len(temp) // 2, 0.0)

    return np.array(temp)