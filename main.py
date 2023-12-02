import torch
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision import transforms
from dataset import VideoDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Instantiate the dataset
dataset = VideoDataset(root_dir='/content/drive/MyDrive/CSVD', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=8, num_workers=1)

# Load the pre-trained MViT2 model
model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)

# Change the number of output features of the model to match the number of classes in your dataset
num_classes = len(dataset.classes)
model.head[1] = torch.nn.Linear(model.head[1].in_features, num_classes)

# Define a loss function
criterion = torch.nn.CrossEntropyLoss()

# Define an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fine-tune the model
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs = inputs.permute(0, 2, 1, 3, 4)

        inputs = inputs.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')

# Print the model architecture
print(model)

# Print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

# Print the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {trainable_params}')

# Or, save only the state dict (recommended)
torch.save(model.state_dict(), 'model_state_dict.pth')
