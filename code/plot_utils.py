import numpy as np
import matplotlib.pyplot as plt




def plot_per_epoch_train(loss_list, start_epoch=0, title=''):
    '''plots training result lists, per epoch, and for each net instance'''
    loss_array = np.asarray(loss_list)
    loss_array = loss_array[start_epoch:,]
    if len(loss_array.shape) > 2: #then print per instance
        for idx in range(loss_array.shape[2]):
            y = np.mean(loss_array[:,:,idx],1)
            error = np.std(loss_array[:,:,idx],1)
            plt.fill_between(range(loss_array.shape[0]), y-error, y+error)
            plt.xlabel('Epoch')
            title_idx = title + ' of instance #' + str(idx)
            plt.title(title_idx)
            plt.show()
    else:
        y = np.mean(loss_array,1)
        error = np.std(loss_array,1)
        plt.fill_between(range(loss_array.shape[0]), y-error, y+error, linestyle='-')
        plt.xlabel('Epoch')
        plt.title(title)
        plt.show()

def plot_per_epoch_test(loss_list, start_epoch=0, title=''):
    '''plots test result lists, per epoch, averaged over all net instances'''
    loss_array = np.asarray(loss_list)
    loss_array = loss_array[start_epoch:,]
    if len(loss_array.shape) > 1: #instances
        loss_array = np.mean(loss_array,1)
    plt.plot(loss_array)
    plt.xlabel('Epoch')
    plt.title(title)
    plt.show()
    

    