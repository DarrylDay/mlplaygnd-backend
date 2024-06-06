import torch
from torch import nn
import mnist.model

model = mnist.model.getNewModel()
print(model)

train_dataloader = mnist.model.getTrainingDataLoader()
test_dataloader = mnist.model.getTestDataLoader()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model.train(train_dataloader, model, loss_fn, optimizer)
    model.test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "models/mnist.pth")
print("Saved PyTorch Model State.")