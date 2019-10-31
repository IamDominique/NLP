import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json






class Classifier():


    def model_setup(self, arch = "resnet50", hidden_units = 1024,learning_rate = 0.001,
                    dropout=0.2, gpu = False):
        '''Methods that builds a model
           Args
           architecture
           number of units of the first hidden layer
           learning rate
           dropout rate
           gpu or cpu
           Returns
           Nothing
        '''
        #initializing
        self.arch = arch
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        #input size and input layer name for the different architectures
        structures = {"vgg16":25088, "resnet50":2048}

        #builing a classifier as per arguments and architecture requirements
        self.classifier = nn.Sequential(nn.Linear(structures[self.arch],hidden_units), # 1st layer
                                        nn.ReLU(),#activation function
                                        nn.Dropout(p=dropout), #adding dropout to avoid overfitting
                                        nn.Linear(hidden_units,hidden_units), # 2nd layer
                                        nn.ReLU(), # activation function
                                        nn.Dropout(p=dropout), #adding dropout to avoid overfitting
                                        nn.Linear(hidden_units,102), # final layer
                                        nn.LogSoftmax(dim = 1)) # final activation function (across colums)


        if self.arch == "resnet50":
            #loading model
            self.model = models.resnet50(pretrained = True)
            #freezing model parameters
            for param in self.model.parameters():
                param.requires_grad = False
            #linking classifier to model
            self.model.fc = self.classifier
            #loss function
            self.criterion = nn.NLLLoss()
            #optimization function and learning rate
            self.optimizer = optim.Adam(self.model.fc.parameters(),learning_rate)


        elif self.arch == "vgg16":
            self.model = models.vgg16(pretrained = True)
            #freezing model parameters
            for param in self.model.parameters():
                param.requires_grad = False
            #linking classifier to model
            self.model.classifier = self.classifier
            #loss function
            self.criterion = nn.NLLLoss()
            #optimization function and learning rate
            self.optimizer = optim.Adam(self.model.classifier.parameters(),learning_rate)


        else :
            print("This model is not supported. Please use vgg16 or resnet50")

        #setting_up device (cpu or gpu)
        if torch.cuda.is_available() and gpu == True:
            self.device = torch.device("cuda")

        else:
            self.device = torch.device("cpu")





    def load_data(self, data_path = "./flowers" ):
        '''
        Loads 3 datasets from the choosen directories and generates 3 dataloader the neural network can use
        Subdirectories for the datasets should be test, train and valid.
        Args
        data path containing the 3 subdirectories
        Returns
        nothing

        '''

        data_dir = data_path
        train_dir = data_dir + '/train/'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'


        #data augmentation and normalization
        train_trans =  transforms.Compose([transforms.Resize(224),
                                           transforms.RandomResizedCrop(size= 224,
                                                                        scale=(0.08, 1.0)),
                                           transforms.RandomRotation(40),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        test_trans = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224), #squaring images
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])


        valid_trans = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),#squaring images
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

        #loading datasets
        self.train_data = datasets.ImageFolder(data_dir + '/train', transform=train_trans)
        self.test_data = datasets.ImageFolder(data_dir + '/test', transform=test_trans)
        self.valid_data = datasets.ImageFolder(data_dir + '/valid', transform = valid_trans)

        #dataloaders
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=32)
        self.validloader = torch.utils.data.DataLoader(self.valid_data, batch_size=20)





    def train(self, epochs = 1):

        """
        Train the model. Prints metric during training
        Args
        training _set : as a torch.utils.data.DataLoader
        validation_set : as a torch.utils.data.DataLoader
        epoch (int) : number of training cycle
        Returns
        Nothing
        """


        #Training Cycle

        print("---------------Training in Progress---------------")
        #Initializing data
        training_loss = 0
        for val in range(epochs):
            #getting data and matching labels from the training_set
            for data, labels in self.trainloader:
                #loading model,data and labels to device
                self.model = self.model.to(self.device)
                data = data.to(self.device)
                labels = labels.to(self.device)
                #reseting gradiants
                self.optimizer.zero_grad()
                #forward pass through the model
                output = self.model(data)
                #delta between labels(real ouput) and output(model output)
                loss = self.criterion(output, labels)
                #backward propagation - how much each layer contributed to the error
                loss.backward()
                #adjusting weights based on backpropagation
                self.optimizer.step()
                #logging training loss for the cycle
                training_loss += loss.item()

        #Validation Cycle

            else:
                #initializing data
                accuracy = 0
                validation_loss = 0
                #turning off gradiants to optimize computations
                with torch.no_grad():
                    #setting model to evaluation mode
                    self.model.eval()
                    #getting data and labels from the dataset
                    for data, labels in self.validloader:
                    #loading model,data and labels to device
                        self.model = model.to(self.device)
                        data = data.to(self.device)
                        labels = labels.to(self.device)
                        #forward pass through the model
                        output = self.model(data)
                        #delta between labels(real ouput) and output(model output)
                        delta = self.criterion(output, labels)
                        validation_loss += delta.item()
                        #converting result from the output layer to percentage
                        percentage = torch.exp(output)
                        #getting top percentages and top classes from the output layer
                        top_percentage, top_class = percentage.topk(1, dim=1)
                        #checking for matches and reshaping labels to a 2d tensor
                        match = top_class == labels.view(*top_class.shape)
                        #calculating accuracy
                        accuracy += torch.mean(match.type(torch.FloatTensor)).item()


             #printing metrics
            print("Epoch {}".format(val +1))
            print("Running Loss {}".format(training_loss / len(validloader)))
            print("validation Loss {}".format(validation_loss / len(validloader)))
            print("Validation Accuracy {}". format(accuracy /len(validloader)))
            #setting training_loss back to 0
            training_loss = 0
            #setting model back to training mode
            model.train()

        print("---------------Training is Complete---------------")



        #saving the number of epoch the model has been trained for
        self.epochs = epochs



    def test_model(self):

        """
        Test the trained model and displaying the accuracy on test batch
        Args
        test_set : as a torch.utils.data.DataLoader
        Returns
        Nothing
        """
        print("---------------Testing in Progress----------------")
        #initalizing data
        accuracy = 0
        #turning off gradiants to optimize computations
        with torch.no_grad():
            #setting model to evaluation mode
            self.model.eval()
            #getting data and labels from the dataset
            for data, labels in self.testloader:
                #loading model,data and labels to device
                self.model = self.model.to(self.device)
                data = data.to(self.device)
                labels = labels.to(self.device)
                #forward pass through the model
                output = self.model(data)
                #delta between labels(real ouput) and output(model output)
                delta = self.criterion(output, labels)
                #converting result from the output layer to percentage
                percentage = torch.exp(output)
                #getting top percentages and top classes from the output layer
                top_percentage, top_class = percentage.topk(1, dim=1)
                #checking for matches and reshaping labels so the equality works
                match = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(match.type(torch.FloatTensor))

        #setting model back to training mode
        self.model.train()

        #printing metrics
        print(f"The model accuracy on the test data is {accuracy /(len(self.testloader))}")
        print("---------------Testing is Complete----------------")



    def save_checkpoint(self,file_name = "checkpoint", save_dir = os.path.abspath("./")):
        """
        Saves the model and the class to index mapping into a checkpoint
        arg :
        checkpoint file name (without extension)
        saving directory

        returns :
        saved checkpoint as a pth file
        """

        #mapping classes to indices using one of the datasets
        self.class_to_idx = self.train_data.class_to_idx

        #building checkpoint
        checkpoint = {
                        'arch': self.arch,
                        'hidden_units' :self.hidden_units,
                        'learning_rate': self.learning_rate,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch': self.epochs,
                        'class_to_idx': self.class_to_idx
                     }
        #formating file name
        file_name = save_dir + "/" + file_name + ".pth"
        print("---------------Saving Checkpoint------------------")

        return torch.save(checkpoint,file_name)


    def load_checkpoint(self, filepath = "checkpoint.pth"):
        """
        Rebuilds the model from a checkpoint
        also reload the matching model.class_to_idx and the
        number of training epochs

        arg :
        path to the checkpoint file


        """
        #initializing classifier
        init = Classifier()
        #loading checkpoint
        checkpoint = torch.load(filepath)

        #rebuilding model
        model = self.model_setup(arch = checkpoint['arch'], hidden_units = checkpoint['hidden_units'],
                    learning_rate = checkpoint['learning_rate'],dropout=0.2, gpu = False)
        #updating model state
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epochs = checkpoint['epoch']
        self.class_to_idx = checkpoint['class_to_idx']
        #freezing model parameter
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def process_image(self,image):
        '''
        Scales, crops, and normalizes an image (using the PIL package)
        The image is turned to a tensor ready to be used by the neural network
        Args :
        image file
        Returns :
        processed image as a tensor

       '''
        buffer = Image.open(image)

        processing = transforms.Compose([transforms.Resize(224), #resizing
                                     transforms.CenterCrop(224), #croping from center
                                     transforms.ToTensor(), #normalization
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        processed_img = processing(buffer)

        return processed_img

    def predict(self,image_path, topk=5, category_names = False, gpu = False):
        ''' Predict the class or classes of an image using a trained deep learning model
            args :
            image path
            number of top predicted classes
            category_names for mapping
            gpu to compute on the gpu if available
            returns:
            nothing, but prints the class or classes predicted together with the probabilities
        '''

        #scale, crops and normalize the image with the process_image function
        img = self.process_image(image_path)
        #adding a batch dimension to avoid runtime error (inplace)
        img = img.unsqueeze_(0)

        #cheking for cpu or gpu use

        if torch.cuda.is_available() and gpu == True:
            self.device = torch.device("cuda")

        else:
            self.device = torch.device("cpu")

        #turning off gradiants to optimize computations
        with torch.no_grad():
            #loading model and img to device
            img = img.to(self.device)
            self.model = self.model.to(self.device)
            #getting model prediction
            output = self.model(img)
            # turning model's output into a probabily
            probabilities = torch.exp(output)
            #getting top percentages and top classes
            top_probabilities, top_classes = probabilities.topk(topk, dim=1)
            #converting top_probabilities to a np.array for data handling and printing
            top_probabilities = np.array(top_probabilities[0])
            #getting idx_to_class into a dictionary for mapping
            idx_to_class = {v : k for k, v in self.class_to_idx.items()}
            top_classes = top_classes[0].tolist()
            #mapping idx to class
            top_classes = [idx_to_class[cls] for cls in (top_classes)]

        if category_names != False:
            with open(category_names, 'r') as json_file:
                cat_to_name = json.load(json_file)
                #mapping class names to class indexes by iterating over the classes numpy array
                names = [cat_to_name[str(cls)] for cls in (top_classes)]
                i=0
                #printing all results
                print(f"---------- Top {topk} Probabilities ----------")
                while i < len(top_probabilities):
                    print("{0:<30}: %{1:2f}".format(names[i], top_probabilities[i]))
                    i += 1

        else :
            i=0
            #printing all results
            print(f"-- Top {topk} Probabilities --")
            while i < len(top_probabilities):
                    print("{0:<14}: %{1:2f}".format(top_classes[i], top_probabilities[i]))
                    i += 1








