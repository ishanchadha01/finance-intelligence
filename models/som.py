import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
 

class SOM(nn.Module):
    """
    2D Self-organizing map with Gaussian kernel similarity function
    and particle swarm optimization for update step
    """
    def __init__(self, x, y, dim, num_iters, learning_radius=.3, sigma=None, cognitive=.1, social=.1, inertia=0.5):
        super(SOM, self).__init__()
        self.x = x
        self.y = y
        self.dim = dim
        self.num_iters = num_iters
        self.learning_radius = learning_radius
        self.cognitive = cognitive
        self.social = social
        self.inertia = inertia

        if sigma is None:
            self.sigma = max(m, n) / 2.0

        self.grid_locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.particles = torch.randn(m*n, dim)
        self.velocities = torch.zeros(m*n, dim)
        self.pdist = nn.PairwiseDistance(p=2)

    def neuron_locations(self):
        # Initialize positions in grid-like arrangement
        for i in range(self.x):
            for j in range(self.y):
                yield np.array([i, j])

    def map_vectors(self, input_vectors):
        output = []
        for vector in input_vectors:
            min_idx = min([i for i in range(len(self.particles))],
                            key=lambda idx: np.linalg.norm(vector-self.particles[idx]))
            output.append(self.grid_locations[min_idx])
        return output

    def particle_swarm_update(self, bmu_idx, neighborhood, radius):
        global_best = self.particles[bmu_idx]
        global_nbest = neighborhood[bmu_idx].float()
        for idx in range(len(self.particles)):
            particle = self.particles[idx]
            velocity = self.velocities[idx]

            # Check if particle is in learning radius
            if global_nbest - neighborhood[idx].float() > radius:
                continue

            # Update each dimension of particle
            for dim in range(self.dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                v_cognitive = self.cognitive * r1 * (global_best[dim] - particle[dim])
                v_social = self.social * r2 * (global_best[dim] - particle[dim])
                v_update = self.inertia * velocity[dim] + v_cognitive + v_social
                self.velocities[idx][dim] = v_update
                p_update = particle[dim] + v_update
                if p_update > 1:
                    p_update = 1
                elif p_update < 0:
                    p_update = 0
                self.particles[idx][dim] = particle[dim] + v_update


    def kernel_func(self, bmu_loc, sigma):
        bmu_distance_squares = torch.sum(torch.pow(self.grid_locations.float() \
            - torch.stack([bmu_loc for i in range(self.x*self.y)]).float(), 2), 1)
        return torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma**2)))

    def forward(self, input, iter_num):

        # Find distance from input vector to each particle, which is location of each particle
        dists = self.pdist(torch.stack([input for i in range(self.x*self.y)]), self.particles)
        
        # Best matching unit is grid location in x by y grid
        _, bmu_idx = torch.min(dists, 0)
        bmu_loc = self.grid_locations[bmu_idx,:]
        bmu_loc = bmu_loc.squeeze()

        # Update learning rate and sigma for time decay
        decay = 1.0 - iter_num/self.num_iters
        lr_decay = self.learning_radius * decay
        sigma_decay = self.sigma * decay

        # Gaussian kernel function for neighborhood function
        neighborhood = self.kernel_func(bmu_loc, sigma_decay)
        
        self.particle_swarm_update(bmu_idx, neighborhood, lr_decay)

        # Update step
        # learning_rate_op = alpha_decay * neighborhood
        # learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1].repeat(self.dim) for i in range(self.x*self.y)])
        # delta = torch.mul(learning_rate_multiplier, (torch.stack([input for i in range(self.x*self.y)]) - self.particles))                                         
        # new_particles = torch.add(self.particles, delta)
        # self.particles = new_particles


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
    n_iter = 10
    som = SOM(m, n, 3, n_iter)
    for iter_no in range(n_iter):
        #Train with each vector one by one
        for i in range(len(data)):
            som(data[i], iter_no)

    #Store a centroid grid for easy retrieval later on
    centroid_grid = [[] for i in range(m)]
    weights = som.particles
    positions = som.grid_locations
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