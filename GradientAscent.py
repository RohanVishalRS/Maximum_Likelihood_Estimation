import math

import numpy as np

from Gibbs import gibbs
from VertexColoringProblem import sumprod

# Perform maximum likelihood estimation to get most likely colors for each node
# of distribution based on empirical evidence samples.
def colormle(A, samples):
    sample_len = len(samples)

    # fetch the unique colors from all samples
    number_of_colors = np.unique(samples)
    color_len = len(number_of_colors)

    # initialize the color vector. Set color of all nodes to 1.
    theta = np.full((1, color_len), 1, dtype=float)

    _, empirical_count = np.unique(samples, return_counts=True)
    empirical_count = empirical_count/sample_len

    # get normalize count of each color for all vertices.
    empirical_count = np.reshape(empirical_count, (1, color_len))
    likelihood = 0
    tolerance = 1e-5

    # get the derivative vectors for each color.
    derivatives = np.ones((1, color_len), dtype=int)
    decayed_learning_rate = learning_rate
    partition = 0
    i = 0
    while True:

        # Use belief propagation to fetch probabilities of all colors for all vertices,
        # and the partition variable (a normalizing constant)
        partition, probabilities = sumprod(np.array(A), theta.transpose(), belief_its)

        belief_probabilities = []

        for probability in probabilities:
            belief_probabilities.append([v for k, v in probability.items()])
        expected_count = np.sum(belief_probabilities, axis = 0)
        derivatives = np.subtract(empirical_count, expected_count.transpose())

        # adjust the color
        theta += decayed_learning_rate * derivatives

        epoch = math.floor(i / 10)

        # give some preference to previous samples by reducing the learning rate after few epochs
        decayed_learning_rate = learning_rate * math.pow(0.5, math.floor((1 + epoch)/10))
        i += 1

        # get the likelihood of the new colors
        new_likelihood = (np.dot(theta, empirical_count.transpose()) - np.log(partition)) * sample_len
        if abs(new_likelihood - likelihood) < tolerance:
            break
        else:
            likelihood = new_likelihood
    return theta, likelihood


if __name__ == '__main__':
    burnin = 1000
    belief_its = 148
    learning_rate = 0.01

    its = [10, 100, 1000, 10000]

    # weights of colors depicting the preferred colors by each node.
    # Higher number means a node would prefer the color more.
    w = np.array([1, 2, 3], dtype=int)

    # Adjacency matrix describing the graph
    A = [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    for it in its:

        # get empirical evidence of distribution of colors on nodes from the gibbs sampler.
        samples = gibbs(A, w, burnin, it)

        # perform Maximum Likelihood estimation to fetch parameter
        weights, likelihood = colormle(A, samples)
        print("likelihood for {} samples is {}".format(it, likelihood))
        print("weights for {} samples is {}".format(it, weights))
        # z, _ = sumprod(np.array(A), weights.transpose(), belief_its)
        # print("Approx partition value for {} samples is {}".format(it, z))
