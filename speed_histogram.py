import pickle as pkl
import matplotlib.pyplot as plt

if __name__=='__main__':

    of_map = pkl.load(open('optical_flow_map.pkl', 'rb'))

    plt.hist(of_map.values(), bins='auto')
    plt.title('Histogram of speeds in the training set.')
    plt.show()