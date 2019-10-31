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
    parser.add_argument("--test_img", type = str,
                        default = os.path.abspath("./flowers/test/1/image_06752.jpg"),
                        help = 'path to the test image')
    parser.add_argument("--checkpoint", type = str, default = os.path.abspath("./checkpoint.pth"),
                        help = 'path to the checkpoint folder')
    parser.add_argument("--top_k", type = int, default = "3",
                        help = 'top X model predictions')
    parser.add_argument("--category_names", dest="category_names", action="store", default = False,
                        help = 'file including category names for mapping')
    parser.add_argument('--gpu', action='store_true',
                         help = "add --gpu to compute on gpu")

    return parser.parse_args()


def main():
    """
    Predict the class or classes of an image using a trained deep learning model
    args :
    path to the test image
    path to the folder where the model checkpoint is. Checkpoint contains all the data to rebuild the trained model
    top X model predictions
    file including category names for mapping
    gpu to compute on the gpu if available
    returns:
    nothing, but prints the class or classes predicted together with the probabilities
    """
    arg = get_input_args()
    #validation
    if os.path.isfile(arg.test_img) == False :
        raise Exception("The image does not exist!")

    init = Classifier()
    init.load_checkpoint(arg.checkpoint)
    init.predict(image_path = arg.test_img, topk= arg.top_k,
                 category_names = arg.category_names, gpu = arg.gpu)




if __name__ == "__main__":
    main()
