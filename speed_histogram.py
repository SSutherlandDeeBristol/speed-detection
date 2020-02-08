import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':

    of_map = pkl.load(open('../optical-flow/train/optical_flow_map.pkl', 'rb'))

    plt.hist(np.concatenate([[s for (_,s) in speeds] for speeds in of_map.values()]), bins='auto')
    plt.title('Histogram of speeds in the training set.')
    plt.show()