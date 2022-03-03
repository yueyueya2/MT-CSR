import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from parameters import *

def cossimi(a, b):
    n = len(a)
    dot = sum([a[i] * b[i] for i in range(n)])
    a_len = np.sqrt(sum([a[i] * a[i] for i in range(n)]))
    b_len = np.sqrt(sum([b[i] * b[i] for i in range(n)]))
    if a_len == 0 or b_len == 0:
        return 0.0
    return dot / (a_len * b_len)

def POI_similarity(POI):
    data = POI
    type_poi = len(data)
    poi = data.reshape(type_poi, -1)
    # calculate the similarity of different region
    poi_similar_index = []
    pearson_similar_value = []
    for i in range(side * side):
        max_pearson = 0.0
        most_similar_region = 0
        for j in range(side * side):
            # avoid to calculate self pearson
            if i == j:
                continue
            poi_a_flag = poi[:, i]
            poi_b_flag = poi[:, j]
            pearsons = cossimi(poi_a_flag, poi_b_flag)
            if np.abs(pearsons) > max_pearson:
                max_pearson = pearsons
                most_similar_region = j
        poi_similar_index.append(int(most_similar_region))
        pearson_similar_value.append(max_pearson)
    return poi_similar_index, pearson_similar_value

