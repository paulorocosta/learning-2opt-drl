import numpy as np
import random
from scipy.spatial import distance_matrix


def create_tour(N):
    """
    Create an initial tour for the TSP

    :param int tour_length: Tour length
    :param bool rand:  Generate random tour

    :return: list with a TSP tour
    """

    tour = random.sample(range(N), N)

    return list(tour)


def calculate_distances(positions):
    """
    Calculate a all distances between poistions

    :param np.array positions: Positions of (tour_len, 2) points
    :return: list with all distances
    """

    # def length(x, y):
    #     return np.linalg.norm(np.asarray(x) - np.asarray(y))
    # distances = [[length(x, y) for y in positions] for x in positions]

    distances = distance_matrix(positions, positions)
    return distances


def route_distance(tour, distances):
    """
    Calculate a tour distance (including 0)

    :param list tour: TSP tour
    :param list : list with all distances
    :return dist: Distance of a tour
    """
    dist = 0
    prev = tour[-1]
    for node in tour:
        dist += distances[int(prev)][int(node)]
        prev = node
    return dist


def swap_2opt(tour, i, k):
    """
    Swaps two edges by reversing a section of nodes

    :param list tour: TSP tour
    :param int i: First index for the swap
    :param int j: Second index for the swap
    """
    # assert tour[0] == 0 and tour[-1] != 0
    if k <= i:
        i_a = i
        i = k
        k = i_a
    assert i >= 0 and i < (len(tour) - 1)
    assert k >= i and k < len(tour)
    new_tour = tour[0:i]
    new_tour = np.append(new_tour, np.flip(tour[i:k + 1], axis=0))
    new_tour = np.append(new_tour, tour[k+1:])
    # assert len(new_tour) == len(tour)
    new_tour = [int(i) for i in new_tour]
    return list(new_tour)


def heuristic_2opt_fi(positions):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    # print("initial distance", best_distance)
    while improvement:
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(best_tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    swap_indices.append([i, k])
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
                    break
            if improvement:
                break
    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    best_tour = np.array(best_tour)
    tours = np.array(tours)
    return best_distance/10000


def heuristic_2opt_bi(positions):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    # print("initial distance", best_distance)
    while improvement:
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(tour, i, k)
                new_distance = route_distance(new_tour, distances)
                # print("i,j", i,k)
                if new_distance < best_distance:
                    swap_indices.append([i, k])
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
        tour = best_tour
    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    best_tour = np.array(best_tour)
    tours = np.array(tours)

    # return best_tour, best_distance/10000

    return best_distance/10000


def heuristic_2opt_fi_restart(positions, steps):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    restart_distance = best_distance
    # print("initial distance", best_distance)
    for n in range(steps):
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
                    tour = new_tour
                    break
            if improvement:
                break
        if improvement is False:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

            tour = create_tour(len(tour))
            best_distance = 1e10
        if n == steps-1:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour
    assert len(best_tour) == len(tour)

    return restart_distance/10000
    # return_dict[procnum] = restart_tour, restart_distance/10000



def heuristic_2opt_bi_restart(positions, steps):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    restart_distance = best_distance
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    # print("initial distance", best_distance)
    for n in range(steps):
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    swap_indices.append([i, k])
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
        tour = best_tour
        if improvement is False:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

            tour = create_tour(len(tour))
            best_distance = 1e10
        if n == steps-1:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return restart_distance/10000
    # return_dict[procnum] = restart_tour, restart_distance/10000
