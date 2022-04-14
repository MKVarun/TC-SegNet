import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from collections import defaultdict
import copy
import time

criterion = nn.CrossEntropyLoss()

def tversky_loss(true, logits, alpha, beta, eps=1e-7):

    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


def jaccard_loss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


def dice_loss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def calc_loss(pred, target, metrics, criterion):
    ce = criterion(pred, target) 
    # tversky = tversky_loss(torch.unsqueeze(target,1), pred)
    iou = jaccard_loss(torch.unsqueeze(target,1), pred)
    dice = dice_loss(torch.unsqueeze(target,1), pred)

        
    metrics['criterion_loss'] += ce.data.cpu().numpy() * target.size(0) # Total Loss
    metrics['dice_loss'] += dice.data.cpu().numpy() * target.size(0)
    metrics['iou_loss'] += iou.data.cpu().numpy() * target.size(0)
    # metrics['tversky'] += tversky.data.cpu().numpy() * target.size(0)
    return metrics

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))  


def train(model, dataloader, criterion, learning_rate, optimizer, num_epochs):
    best_loss = 1e10
    mean_train_losses = []
    mean_val_losses = []
    plt_loss = {'dice_loss':[], 'iou_loss':[],'criterion_loss':[]}

    # criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
      
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)
      since = time.time()

      for phase in ['train', 'val']:
          if phase == 'train':
            print("BEGIN TRAIN")
            model.train()  # Set model to training mode
          else:
            print("BEGIN EVAL")
            model.eval()   # Set model to evaluate mode

          metrics = defaultdict(float)
          epoch_samples = 0  
     
          for images, masks in dataloader[phase]:  
                    
              images = Variable(images.cuda())
              masks = Variable(masks.cuda())
              
              optimizer.zero_grad()

              with torch.set_grad_enabled(phase == 'train'): # If false grads are not computed for validation
                  outputs = model(images)
                  m_loss = calc_loss(outputs, masks, metrics, criterion) # Batch Loss i.e loss calculated for 16 samples.
                  loss = criterion(outputs, masks)

                  if phase == 'train':
                      # train_losses.append(loss.data)
                      loss.backward()
                      optimizer.step()

              epoch_samples += images.size(0)  # Adding batch size - To compute total samples in a batch

          print_metrics(metrics, epoch_samples, phase)
          epoch_loss = metrics['dice_loss'] / epoch_samples
          plt_loss['dice_loss'].append(epoch_loss)
          plt_loss['iou_loss'].append(metrics['iou_loss'] / epoch_samples)
          plt_loss['criterion_loss'].append(metrics['criterion_loss'] / epoch_samples)
              

          if phase == 'val' and epoch_loss < best_loss:
              print("saving best model")
              
              best_loss = epoch_loss
              best_model_wts = copy.deepcopy(model.state_dict())

      time_elapsed = time.time() - since
      print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model,plt_loss