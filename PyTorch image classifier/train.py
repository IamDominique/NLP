import os
import argparse
from image_classifier import Classifier


def get_input_args():

    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to create and define these command line arguments. If
    the user fails to provide some or all of the arguments, then the default
    values are used for the missing arguments.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type = str, default = os.path.abspath("./"),
                        help = 'path to the checkpoint folder')
    parser.add_argument("--data_dir", type = str, default = os.path.abspath("./flowers"),
                        help = 'path to the data folder')
    parser.add_argument("--arch", type = str, default = "resnet50",
                        help = 'NN model Architecture : vgg16 or resnet50')
    parser.add_argument("--hidden_units", type = int, default = 1024,
                         help = "size of the first hidden layer for the classifier")
    parser.add_argument("--epochs", type = int, default = 10,
                         help = "number of training epochs")
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                         help = "model learning rate")
    parser.add_argument('--gpu', action='store_true',
                         help = "add --gpu to compute on gpu rather than on cpu")

    return parser.parse_args()





def main():
    """
    Build and train a new network on a dataset and save the model as a checkpoint
    Args
    path to the checkpoint saving folder
    path to the training data folder
    choosen architecture (resnet50 or vgg16)
    size of the first hidden layer for the classifier
    number of training epochs
    model learning rate
    gpu : computation on gpu
    Returns
    checkpoint as a pth.file
    """
    arg = get_input_args()
    #validation
    if os.path.isdir(arg.data_dir) == False :
        raise Exception("The image directory does not exist!")
    if {'test','train','valid'}.issubset((os.listdir(arg.data_dir))) == False:
        raise Exception('Missing test, train or valid sub-directories!')
    #building, training and saving
    init = Classifier()
    init.model_setup(arch = arg.arch, hidden_units = arg.hidden_units,
                     learning_rate = arg.learning_rate, dropout = 0.2, gpu = arg.gpu)
    init.load_data(arg.data_dir)
    init.train(arg.epochs)
    init.test_model()
    init.save_checkpoint(file_name = "checkpoint", save_dir = arg.save_dir)


if __name__ == "__main__":
    main()




