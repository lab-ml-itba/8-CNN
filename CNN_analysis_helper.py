import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def get_FASHION_MNIST_data(folder, test_split = 0.2):
    x = np.load(folder+'train_images.npy')
    y = np.loadtxt(folder+'train_labels.csv', delimiter=',', skiprows=1)
    x_test_ = np.load(folder+'test_images.npy')
    y_test = np.loadtxt('test_labels.csv', delimiter=',', skiprows=1)
    x_train_, x_valid_, y_train, y_valid = train_test_split(x, y, test_size = test_split)
    x_train = x_train_.reshape(x_train_.shape + (1,))
    x_valid = x_valid_.reshape(x_valid_.shape + (1,))
    x_test = x_test_.reshape(x_test_.shape + (1,))

    y_train_categorical = to_categorical(y_train)
    y_val_categorical = to_categorical(y_valid)
    y_test_categorical = to_categorical(y_test[:,0])
    return x_train, x_valid, y_train_categorical, y_val_categorical

def generate_random_image(std_dev=10, img_width=28, img_height=28):
    input_img_data = (np.random.normal(0, std_dev, (1,img_width, img_height, 1))) + 128.
    return input_img_data

def plot_filter_coefs(layer_name, model, starting_filt = 0, max_filters = 6, normalize = True):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    n_filters = min(layer_dict[layer_name].filters, max_filters)
    f, ax = plt.subplots(1, n_filters, figsize=(20,8))
    if type(ax) is not np.ndarray:
        ax = [ax]
    if normalize:
        min_value = np.min(layer_dict[layer_name].get_weights()[0])
        fil_w_norm = layer_dict[layer_name].get_weights()[0] - min_value
        max_value = np.max(fil_w_norm)
        fil_w_norm = (255*fil_w_norm/max_value).astype(int)
    else:
        fil_w_norm = layer_dict[layer_name].get_weights()[0]
    
    for j in range(n_filters):
        i = j + starting_filt
        filter_weights = fil_w_norm[:,:,:,i][:,:,0]
        #filter_weights = filter_weights - np.min(filter_weights)
        #filter_weights = filter_weights/np.max(filter_weights)
        for row in range(filter_weights.shape[0]):
            for col in range(filter_weights.shape[1]):
                coded_weight = filter_weights[col,row]
                ax[j].text(row, col, coded_weight, horizontalalignment='center', backgroundcolor='white')
        ax[j].axis('off')
        if normalize:
            ax[j].imshow(filter_weights, cmap='gray', vmin=0, vmax= 255)
        else:
            ax[j].imshow(filter_weights, cmap='gray')
    plt.show()
    

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # x = x.transpose((1, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def do_gradient_ascent(model, layer_name, filter_index, input_img_data_raw, step = 1, deprocess_image_flag=True, iterations = 20, img_width = 28, img_height = 28):
    global grad_glob
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if type(layer_dict[layer_name]) == keras.layers.convolutional.Conv2D:
        loss = K.mean(layer_output[:, :, :, filter_index])
    else:
        loss = K.mean(model.output[:, filter_index])
    input_img = model.input

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    # https://stats.stackexchange.com/questions/22568/difference-in-using-normalized-gradient-and-gradient
    grads = normalize(grads)
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])
    
    # run gradient ascent for 20 steps
    input_img_data = input_img_data_raw.copy()
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        grad_glob = grads_value
        #print(grads_value.shape)
        input_img_data += grads_value * step
        #step = step*0.99
        
    if deprocess_image_flag:
        img_raw = input_img_data[0].reshape(img_width, img_height)
        img = deprocess_image(img_raw)
        return img
    else:
        return input_img_data
    
def plot_conv_filters(model, layer_name, input_img_data, iterations=200, step = 1):
    # filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer
    # we start from a gray image with some noise
    # ax[0].imshow(input_img_data.reshape(img_width, img_height) , cmap='gray')
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if type(layer_dict[layer_name]) == keras.layers.convolutional.Conv2D:
        filters = layer_dict[layer_name].filters
    elif type(layer_dict[layer_name]) == keras.layers.core.Dense:
        filters = layer_dict[layer_name].output_shape[-1]
    else:
        print('Not supported layer')
        return None
    filt_x = np.min([filters, 6])
    filt_y = int(np.ceil(filters / 6))
    f, ax = plt.subplots(filt_y, filt_x, figsize=(20,3*filt_y))
    ax = ax.reshape(-1)
    images = []
    for i in range(filters):
        img = do_gradient_ascent(model, layer_name, i, input_img_data, iterations=iterations, step=step)
        images.append(img)
        ax[i].axis('off')
        ax[i].imshow(img, cmap='gray')
    plt.show()
    return images

def plot_activations(activations_output, relative=False, n_filters=6):
    f, ax = plt.subplots(activations_output.shape[0], n_filters, figsize=(10,6))
    f.tight_layout()
    for j in range(activations_output.shape[0]):
        vmax = activations_output[j,:,:,:].max()
        vmin = activations_output[j,:,:,:].min()
        
        #f.tight_layout() 
        for i in range(n_filters):
            ax[j, i].axis('off')
            if relative:
                ax[j, i].imshow(activations_output[j,:,:,i], cmap='gray', vmax=vmax, vmin=vmin)
            else:
                ax[j, i].imshow(activations_output[j,:,:,i], cmap='gray')
    plt.show()
    
def select_images_from_dataset(imagesIdxs = [0,53,5,50,9]):
    images_to_filter = x_train[imagesIdxs[0]][:,:,0].T
    for i in imagesIdxs[1:]:
        images_to_filter = np.dstack((images_to_filter, x_train[i][:,:,0].T))
    images_to_filter = images_to_filter.reshape(5, 28,28,1)
    return images_to_filter