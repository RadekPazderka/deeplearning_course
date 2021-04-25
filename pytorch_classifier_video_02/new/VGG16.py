import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from pytorch_classifier_video_02.new.model import vgg16

BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 50
N_CLASSES = 25




transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

trainData = dsets.ImageFolder(r'C:\Users\darkwolf\PycharmProjects\deeplearning_course\caffe_classifier_video_02\dataset\data\TRAIN', transform) #TODO: bere presne strukturu kterou mam pripravenou, podslozky jsou labely a v nich obrazky dane tridy
testData = dsets.ImageFolder(r'C:\Users\darkwolf\PycharmProjects\deeplearning_course\caffe_classifier_video_02\dataset\data\val', transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)




# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Train the model
for epoch in range(EPOCH):

    avg_loss = 0
    cnt = 0
    for images, labels in trainLoader:
#        images = images.cuda()
#        labels = labels.cuda()
#
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, outputs = vgg16(images)
        loss = cost(outputs, labels)
        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)

# Test the model
vgg16.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = images.cuda()
    _, outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(predicted, labels, correct, total)
    print("avg acc: %f" % (100 * correct / total))

# Save the Trained Model
torch.save(vgg16.state_dict(), 'vgg16.pkl')