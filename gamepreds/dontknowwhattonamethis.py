import cv2
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
# x is the long dimension

def closest_pocket(ball):
    d = 2000
    for pocket in [(0,0),(790,0),(1580,0),(0,790),(790,790),(1580,790)]:
        d_p = np.sqrt((ball[1]-d[0])**2+(ball[2]-d[1])**2)
        d = d_p if d_p < p else d_p
    return d

# 0 for easy, 1 for med, 2 for hard
def zone(ball):
    d = closest_pocket(ball)
    if d < 20:
        return 0
    elif d < 50:
        return 1
    else:
        return 2

# input: [#easy solid, med solid, hard solid, easy stripe, med stripe, hard stripe]
def ball_to_input(balls):
    input = np.zeros(6)
    for ball in balls:
        z = zone(ball)
        input[z + (balls[0] == 'stripe')] += 1
    return input

# Neural Network Model (1 hidden layer)
########## MAKE THIS LOGISTIC!!!!!!!!!!!!! ##################
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Hyper Parameters
input_size = 6
hidden_size = 12
num_classes = 2
num_epochs = 10
batch_size = 1
learning_rate = 0.001

net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

# TODO: make train_data a list of [states, winners]
# or figure out some other way


# Train the Model
for epoch in range(num_epochs):
    for states, winners in train_data: 
        # Convert torch tensor to Variable
        states = Variable(states.view(-1, input_size))
        winners = Variable(winners)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(states)
        loss = criterion(outputs, winners)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for states, winners in test_data:
    states = Variable(states.view(-1, input_size))
    outputs = net(states)
    _, predicted = torch.max(outputs.data, 1)
    total += winners.size(0)
    correct += (predicted == winners).sum()

# print('Accuracy of the network on the 100 test states: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')