import numpy as np

def gaussuian_filter(kernel_size, sigma= (1,1), muu= (0,0)):
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size[0]),
                       np.linspace(-1, 1, kernel_size[1]))

    prob = np.exp(-1* (  ((x-muu[0])/sigma[0])**2 + ((y-muu[1])/sigma[1])**2))
    return prob



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    filter = gaussuian_filter((80,40))
    plt.imshow(filter)
    plt.show()