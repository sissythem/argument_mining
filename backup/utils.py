import torch


def convert_ndarray_to_tensor(array, device_name):
    print("This is horribly inefficient -- TODO fix")
    tensors = []
    for ar in array:
        dat = torch.LongTensor([ar]).to(device_name)
        tensors.append(dat)
    return tensors

def get_accuracy_numcorrect(preds, y, is_proba=True):
    if is_proba:
        # NOTE: this is applicable only for single-label classification
        # for multiclass, you have to use a threshold, e.g. thresh=0.5:
        # predictions = (output > thresh)
        # and modify the code below for multiclass stats and metrics
        preds = torch.argmax(preds, dim=1)
    correct_boolean = (y == preds)
    accuracy = correct_boolean.float().mean().cpu()
    num_correct = correct_boolean.sum().cpu()
    return accuracy, num_correct
