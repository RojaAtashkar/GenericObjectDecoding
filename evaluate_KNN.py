import numpy as np
import torch
from torch import nn

'''
functions from https://github.com/matteoferrante/semantic-brain-decoding
'''


def val_brain_epoch(model, val_dataloader, criterion=None, optim=None, device="cpu"):
    """
    Validates the model using the specified dataloader.

    Parameters:
    - model: PyTorch model to be validated
    - val_dataloader: Dataloader for the validation data
    - criterion: Loss function to be used for validation
    - optim: Optimizer to be used for validation (not used in this function)
    - device: Device to run the model and data on (CPU or GPU)

    Returns:
    - mean_loss: Mean loss of the model over the validation data
    """
    model.eval()

    loss_tmp = []
    i = 0
    # Iterate over the validation data
    with torch.no_grad():
        for x, y in val_dataloader:
            # Move the data to the specified device
            x, y = x.to(device), y.to(device)
            # Get the model's prediction for the input data
            y_pred = model(x)
            # Calculate the loss based on the criterion and model's prediction
            if isinstance(criterion, nn.CosineEmbeddingLoss):
                # Set the target to 1 if using the CosineEmbeddingLoss criterion
                target = torch.ones(y_pred.shape[0]).to(device)
                loss = criterion(y_pred.squeeze(), y.squeeze(), target)
            else:
                loss = criterion(y_pred.squeeze(), y.squeeze())
            # Add the loss to the list of losses
            loss_tmp.append(loss.item())
    # Calculate the mean loss over all the data
    mean_loss = np.mean(loss_tmp)
    return mean_loss


def evaluate_topk(brain, model, k1=5, k2=5):
    """This function takes as input two sets of predictions (brain and model)
    and two integers (k1 and k2) representing the number of top predictions
    to consider for each set. It returns a dictionary of the proportion of
    predictions for each overlap between the top-k predictions from the brain model and the image model"""

    # Initialize a dictionary to store the accuracy for each overlap of the top-k predictions from the brain and
    # model models
    acc = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Iterate over the predictions for each sample
    for i in range(len(brain)):
        # Get the unique indices of the top-k predictions from the brain model and the top-k predictions from the
        # model model
        n_un = len(torch.cat((torch.topk(brain[i], 5).indices, torch.topk(model[i], 5).indices)).unique())
        # Calculate the overlap between the top-k predictions from the brain model and the top-k predictions from the
        # model model
        n_overlap = int(k1 + k2) - n_un

        # Increment the accuracy for the current overlap
        acc[n_overlap] += 1

    # Calculate the proportion of predictions for each overlap
    for k in acc.keys():
        acc[k] /= len(brain)
    return acc


def evaluate_topk_nbrs(brain, model):
    """
    This function compares the labels predicted by two models (brain and model)
    on a set of samples and counts the number of samples with a certain number
    of overlapping labels. It does this by iterating over the samples, comparing
    the labels element-wise, and counting the number of labels that are present in both sets.
    The counts are stored in a dictionary, where the keys are the number of overlapping
    labels and the values are the number of samples with that number of overlapping labels.
    The function then normalizes the counts by the number of samples and returns the resulting
    dictionary.
    """

    # Initialize a dictionary to store the number of samples with a certain number of overlapping labels
    acc = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Iterate over the samples
    for i in range(len(brain)):
        # Get the labels for the current sample from the brain and model
        t1 = brain[i]
        t2 = model[i]

        # Repeat the model labels, so they can be compared element-wise with the brain labels
        compareview = t2.repeat(t1.shape[0], 1).T

        # Find the labels that are present in both the brain and model
        intersection = t1[(compareview == t1).T.sum(1) == 1]

        # Count the number of overlapping labels
        n_overlap = len(intersection)

        # Increment the count for the corresponding key in the dictionary
        acc[n_overlap] += 1

    # Normalize the counts by the number of samples
    for k in acc.keys():
        acc[k] /= len(brain)

    # Return the dictionary
    return acc


def top_5_acc(brain, model, k=5):
    """This function takes as input two sets of predictions (brain and model)
    and an integer (k) representing the number of top predictions to consider.
    It returns the proportion of predictions for which the top prediction
    from the model is among the top-k predictions from the brain model.
    The brain and model variables contain the predicted class probabilities
     for each sample, and the top_5_acc function compares the top prediction
     from the model model with the top-k predictions from the brain model to determine if they match.
    """
    # Initialize a variable to store the accuracy
    acc = 0

    # Iterate over the predictions for each sample
    for i in range(len(brain)):
        # Check if the top prediction from the model model is among the top-k predictions from the brain model
        if model[i].argmax() in torch.topk(brain[i], k).indices:
            # If it is, increment the accuracy
            acc += 1
    # Calculate and return the proportion of correct top-k predictions
    return acc / len(brain)


def top_5_nbrs_acc(brain, model, nbrs_model_train=None):
    """This function takes as input two sets of predictions (brain and model).
    It returns the proportion of predictions for which the label with the highest
     count among the top-5 nearest neighbors of the model model's prediction is
     among the top-5 predictions from the brain model.
     The brain variable contains the top-5 predictions
     from the brain model and the model variable contains the labels of
     the top-5 nearest neighbors of the model model's prediction.
     The top_5_nbrs_acc function compares the label with the highest count
     among the top-5 nearest neighbors of the model model's prediction
     with the top-5 predictions from the brain model to determine if they match.
    """

    # Initialize a variable to store the accuracy
    acc = 0

    # Iterate over the predictions for each sample
    for i in range(len(brain)):
        # Get the unique labels and counts of the top-5 nearest neighbors from the model model
        values, counts = nbrs_model_train[i].unique(return_counts=True)
        # Find the label with the highest count among the top-5 nearest neighbors
        knn = values[counts.argmax()]

        # Check if the label with the highest count among the top-5 nearest neighbors is among the top-5 predictions
        # from the brain model
        if knn in brain[i]:
            # If it is, increment the accuracy
            acc += 1
    # Calculate and return the proportion of correct top-5 nearest neighbor predictions
    return acc / len(brain)
