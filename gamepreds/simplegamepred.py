import cv2
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from ../readercleaner import get_data1
# x is the long dimension

# To change to 6 features, simply change input_size and get_data() call
# To change to "variable length" NN, it's the same idea
# To change to logistic regression, only have a fc and softmax layer

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # i don't need Softmax here, CE loss takes care of it
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyper Parameters
input_size = 2
hidden_size = 12
num_classes = 1 # ?
num_epochs = 10
batch_size = 1
learning_rate = 0.001

net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# change all this
data = get_data1(0,4) # just get 4 games for now
train_data = data.iloc[:4]
test_data = data.iloc[4:]

# Train the Model
for epoch in range(num_epochs):
    for i in range(0, train_data.shape[0], batch_size): 
        batch = train_data.iloc[i:i+batch_size]
        # states and winners tensors
        states = torch.Tensor(batch.loc[:,batch.columns != 'winner'].as_matrix())
        winners = torch.Tensor(batch['winner'].as_matrix())

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
for i in range(0, test_data.shape[0], batch_size):
    batch = test_data.iloc[i:i+batch_size]
    # states and winners tensors
    states = torch.Tensor(batch.loc[:,batch.columns != 'winner'].as_matrix())
    winners = torch.Tensor(batch['winner'].as_matrix())

    states = Variable(states.view(-1, input_size))
    outputs = net(states)
    _, predicted = torch.max(outputs.data, 1)
    total += winners.size(0)
    correct += (predicted == winners).sum()

# print('Accuracy of the network on the 100 test states: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')
