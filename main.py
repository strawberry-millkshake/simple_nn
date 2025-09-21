from nn import nn
from data_example import handwritten

if __name__ == "__main__":

    network = nn()
    network.train_and_run(handwritten, handwritten, 100000)