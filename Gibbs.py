import numpy as np
import random
import copy


# Randomly choose a valid color for a given vertex based on the colorings of
# neighbours from previous samples
def get_sample(A, w, vertex, prev_sample):
    # get probabilities of each color for vertex.
    probabilities = compute_probability_for_vertex(A, w, vertex, prev_sample)

    # choose a random number between 0 and 1
    random_no = random.uniform(0, 1)
    color = 0
    running_sum = 0

    for ind in range(len(probabilities)):

        # randomly select one color based on random_no
        curr_sum = running_sum + probabilities[ind]
        if running_sum < random_no <= curr_sum:
            color = ind
            break
        running_sum = curr_sum
    return w[color]


# fetch an initial color scheme for the graph.
def initial_assignment(A, w):
    num_vertices = len(A)
    assignment = [0] * num_vertices
    assignment[0] = w[0]
    for vertex in range(1, num_vertices):
        for color in w:
            neighbours = fetch_neighbours(A, vertex)
            is_safe = True
            for neighbour in neighbours:
                if assignment[neighbour] == color:
                    is_safe = False
            if is_safe:
                assignment[vertex] = color
                break
    return assignment


# Sampler to fetch an empirical distribution of colors on nodes of graph that
# approximates the true solutions for the graph coloring problem.
# A is the graph
# w is the list of all colors and their weights
# burnin discard all samples in the burnin period
def gibbs(A, w, burnin, its):
    # define the size of samples to store for further processing
    samples = np.zeros((its, len(A)), dtype=int)

    # initialize the graph nodes with random colors.
    assignment = initial_assignment(A, w)
    for i in range(its + burnin):
        # store samples for computing probabilities after discarding burnin amount of samples
        sample_t = copy.deepcopy(assignment)
        vertices_count = len(A)
        for j in range(vertices_count):
            # get next valid sample from gibbs sampler
            sample = get_sample(A, w, j, sample_t)
            sample_t[j] = int(sample)

            # discard samples within the burnin threshold.
        if i >= burnin:
            samples[i - burnin] = sample_t
        assignment = sample_t
    return samples


# checks the probability of a given color for vertex. If neighbouring nodes have same color,
# set the probability of the color for the selected node to 0.
# A is the graph on which nodes are colored
# w is the list of all colors and their weights.
# vertex is a node of graph A
# prev_sample is graph nodes with colors assigned to them in previous iteration.
def compute_probability_for_vertex(A, w, vertex, prev_sample):
    weights = np.exp(w)
    neighbours = fetch_neighbours(A, vertex)
    for ind in range(len(w)):
        for neighbour in neighbours:
            if w[ind] == prev_sample[neighbour]:
                weights[ind] = 0
    probabilities = weights / np.sum(weights)
    return probabilities


# Fetches Neighbouring nodes of a vertex in graph.
def fetch_neighbours(A, ind):
    edges_for_vertex = A[ind]
    indexes = np.nonzero(edges_for_vertex)[0]
    return indexes


# get probabilities of all colors for each vertex in graph
def get_marginals(samples, w):
    size = samples.shape[0]
    unique_count = [None] * samples.shape[1]
    normalised_d = [None] * samples.shape[1]
    for i in range(samples.shape[1]):
        unique_count[i] = {}
        for j in range(len(w)):
            unique_count[i][w[j]] = 0

    for sample in samples:
        for vertex in range(len(sample)):
            color = sample[vertex]
            unique_count[vertex][color] += 1
    sum(unique_count[0].values())

    for vertex in range(len(unique_count)):
        colors = unique_count[vertex]
        normalised_d[vertex] = {k: v / size for k, v in colors.items()}

    return normalised_d


# print the probabilities of colors for each nodes
def print_marginals(marginals, it, burnin, color=None, vertex=None):
    if vertex == None:
        for vertex in range(len(marginals)):
            print("Marginals for vertex {} in {} iterations with {} burnin".format(vertex, it, burnin))
            if color == None:
                for color, probability in marginals[vertex].items():
                    print("Marginals of color {} for vertex {} is {} in {} iterations with {} burnin".format(color,
                                                                                                             vertex,
                                                                                                             probability,
                                                                                                             it,
                                                                                                             burnin))
            else:
                print("Marginals of color {} for vertex {} is {} in {} iterations with {} burnin".format(color, vertex,
                                                                                                         marginals[
                                                                                                             vertex][
                                                                                                             color], it,
                                                                                                         burnin))
    else:
        if color == None:
            for color, probability in marginals[vertex].items():
                print(
                    "Marginals of color {} for vertex {} is {} in {} iterations with {} burnin".format(color, vertex,
                                                                                                       probability,
                                                                                                       it, burnin))
        else:
            print("Marginals of color {} for vertex {} is {} in {} iterations with {} burnin".format(color, vertex,
                                                                                                     marginals[
                                                                                                         vertex][
                                                                                                         color], it,
                                                                                                     burnin))


def get_initial_assignment():
    # ind 0 is A, ind 1 is B, ind 2 is C, ind 3 is D
    return np.array([4, 3, 3, 2], dtype=int)


if __name__ == '__main__':
    its = [2 ** 6, 2 ** 10, 2 ** 12]
    burnin = 1024
    # Adjacent matrix: Represents the connectivity of nodes in the graph
    A = [[0, 1, 1, 1],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [1, 1, 1, 0]]

    # weights of colors: represents which color should be preferred more.
    w = np.array([1, 2, 3, 4], dtype=int)

    for it in its:
        for burnin in its:
            samples = gibbs(A, w, burnin, it)
            marginals = get_marginals(samples, w)
            print_marginals(marginals, it, burnin, 4, 0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
