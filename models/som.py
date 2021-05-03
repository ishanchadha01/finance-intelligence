import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
 

class SOM(nn.Module):
    """
    2D Self-organizing map with Gaussian kernel similarity function
    and particle swarm optimization for update step
    """
    def __init__(self, x, y, dim, num_iters, lr=.3, sigma=None):
        super(SOM, self).__init__()
        self.x = x
        self.y = y
        self.dim = dim
        self.num_iters = num_iters
        self.lr = lr

        if sigma is None:
            self.sigma = max(m, n) / 2.0

        self.weights = torch.randn(m*n, dim)
        self.positions = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.velocities = torch.LongTensor(np.array(list(self.neuron_velocities())))
        self.pdist = nn.PairwiseDistance(p=2)

    def neuron_locations(self):
        # Initialize positions in grid-like arrangement
        for i in range(self.x):
            for j in range(self.y):
                yield np.array([i, j])

    def neuron_velocities(self):
        # Initialize velocities facing outward from center
        for i in range(self.x):
            for j in range(self.y):
                yield np.array([i - np.floor(self.x/2), j - np.floor(self.y/2)])

    def map_vectors(self, input_vectors):
        output = []
        for vector in input_vectors:
            min_idx = min([i for i in range(len(self.weights))],
                            key=lambda idx: np.linalg.norm(vector-self.weights[idx]))
            output.append(self.positions[min_idx])
        return output

    def forward(self, x, it):
        dists = self.pdist(torch.stack([x for i in range(self.x*self.y)]), self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.positions[bmu_index,:]
        bmu_loc = bmu_loc.squeeze()
        
        learning_rate_op = 1.0 - it/self.num_iters
        alpha_op = self.lr * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        bmu_distance_squares = torch.sum(torch.pow(self.positions.float() - torch.stack([bmu_loc for i in range(self.x*self.y)]).float(), 2), 1)
        
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
        
        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1].repeat(self.dim) for i in range(self.x*self.y)])
        delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self.x*self.y)]) - self.weights))                                         
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights


if __name__=='__main__':
    m = 20
    n = 30

    #Training inputs for RGBcolors
    colors = np.array(
        [[0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0.5],
        [0.125, 0.529, 1.0],
        [0.33, 0.4, 0.67],
        [0.6, 0.5, 1.0],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
        [.33, .33, .33],
        [.5, .5, .5],
        [.66, .66, .66]])
    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
        'greyblue', 'lilac', 'green', 'red',
        'cyan', 'violet', 'yellow', 'white',
        'darkgrey', 'mediumgrey', 'lightgrey']

    data = list()
    for i in range(colors.shape[0]):
        data.append(torch.FloatTensor(colors[i,:]))
    
    #Train a 20x30 SOM with 100 iterations
    n_iter = 100
    som = SOM(m, n, 3, n_iter)
    for iter_no in range(n_iter):
        #Train with each vector one by one
        for i in range(len(data)):
            som(data[i], iter_no)

    #Store a centroid grid for easy retrieval later on
    centroid_grid = [[] for i in range(m)]
    weights = som.weights
    positions = som.positions
    for i, loc in enumerate(positions):
        centroid_grid[loc[0]].append(weights[i].numpy())
    
    #Get output grid
    image_grid = centroid_grid

    #Map colours to their closest neurons
    mapped = som.map_vectors(torch.Tensor(colors))

    #Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()