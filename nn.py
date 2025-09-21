import numpy as np
from matplotlib import pyplot as plt
from data_example import display_digit

class nn():

    def __init__(self):

        #the size of the layers
        self.size_input = 20
        self.size_hidden = 10
        self.size_output = 10

        self.error_trace = []

        #create the layers
        self.input = np.zeros(self.size_input)
        self.hidden = np.zeros(self.size_hidden)
        self.observed_output = np.zeros(self.size_output)

        #create the weights (random number between +1/3 and -1/3. 1/3 just seemed to work well)
        self.weights_input_to_hidden = (np.random.rand(self.size_input,self.size_hidden) - 0.5) / 1.5
        self.weights_hidden_to_output = (np.random.rand(self.size_hidden,self.size_output) - 0.5) / 1.5

        # using keyword delta for derivative
        self.delta_input_to_hidden = np.copy(self.weights_input_to_hidden)
        self.delta_hidden_to_output = np.copy(self.weights_hidden_to_output)

        #"old" variables are used for momentum in back prop
        self.old_delta_input_to_hidden = np.copy(self.weights_input_to_hidden)
        self.old_delta_hidden_to_output = np.copy(self.weights_hidden_to_output)

        self.delta_output = self.observed_output
        self.delta_input_to_hidden = self.hidden
        self.alpha = 0.4
    
    def sigmoid(self, input):
        return 1/(1+np.exp(-1 * input))
    
    def train(self, training_data, training_length):

        epoch = 0
        gain = 0

        #main while
        while(gain < 98 and epoch < training_length):
            epoch += 1
            gain = 0

            for k in range(10):

                expected_output = np.zeros(10)
                expected_output[k] = 1
                self.input = training_data[k]

                # ---- forward pass ----

                #input to hidden
                self.hidden = self.sigmoid(np.dot(np.transpose(self.weights_input_to_hidden), self.input))
                
                #hidden to output
                self.observed_output = self.sigmoid(np.dot(np.transpose(self.weights_hidden_to_output), self.hidden))

                # ---- backwards pass ----
                
                #calculate the derivative of the output loss
                self.delta_output = (expected_output - self.observed_output) * self.observed_output * (1-self.observed_output) # basic loss * sig_dirv. sig_dirv = x*(1-x)

                #update the hidden -> output weights
                self.delta_hidden_to_output = self.alpha * self.delta_output * np.transpose(np.tile(self.hidden,(len(self.observed_output),1))) #fill in the delta matrix with the derivative of the output loss times the activation of the hidden layer
                self.weights_hidden_to_output += self.delta_hidden_to_output + (self.old_delta_hidden_to_output*0.5) #update the weights plus the momentum
                self.old_delta_hidden_to_output = self.delta_hidden_to_output #save the current derivatives to do momentum in the next round of backprop

                #calculate the derivative of the hidden loss
                self.delta_input_to_hidden =  np.dot(self.delta_output, self.weights_hidden_to_output) * self.hidden * (1-self.hidden) #just the weights times the derivative at the previous layer times the derivative of the activation

                #update the input -> hidden weights
                self.delta_input_to_hidden = self.alpha * self.delta_input_to_hidden * np.transpose(np.tile(self.input,(len(self.hidden),1))) #same as for the previous layers weights
                self.weights_input_to_hidden += self.delta_input_to_hidden + (self.old_delta_input_to_hidden*0.5)
                self.old_delta_input_to_hidden = self.delta_input_to_hidden

                #calculate the derivative of the loss
                gain += np.dot(((2*self.observed_output)-1), ((2*expected_output)-1)) #derivative of squared loss
            
            if (epoch % 500) == 0:
                print("gain = ", gain, " epoch = ", epoch)
            
            self.error_trace.append(gain)

    def plot_gain(self):
        plt.plot(self.error_trace)
        plt.savefig("graphs/gain_graph")

    def test(self, test_data):

        for k in range(10):

            expected_output = np.zeros(10)
            expected_output[k] = 1
            self.input = test_data[k]

            #input to hidden
            self.hidden = self.sigmoid(np.dot(np.transpose(self.weights_input_to_hidden), self.input))
            
            #hidden to output
            self.observed_output = self.sigmoid(np.dot(np.transpose(self.weights_hidden_to_output), self.hidden))

            print("when showing: ")
            display_digit(test_data[k])
            print("the activation in the hidden layer: ")
            print("expected output", expected_output)
            print("network output", np.round(self.observed_output,2))

    def train_and_run(self,training_data, test_data, training_length):
        self.train(training_data, training_length)
        self.test(test_data)
        self.plot_gain()