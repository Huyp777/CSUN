import logging
import time
import os
from tqdm import tqdm
import torch
from .tools import evaluate

mu = 0.015
alpha = 0.02

def predict(fine_model, coarse_model, data_loader, device, timer=None):
    fine_model.eval()
    coarse_model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for batch in tqdm(data_loader):
        batches, targets, idxs = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            output = fine_model(batches.to(device))
            check = coarse_model(batches.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()

            output = [o.to(cpu_device) for o in output]

        for i, (img_id, result) in enumerate(zip(idxs, output)):
            score = (check[0, i, -1] - check[0, i, 31]) / check[0, i, 31]
            if score > alpha:
                result[-2][-1] = abs(result[-2][-1])*10
            elif score >= mu:
                result[1:-3][-1] = abs(result[1:-3][-1])*10
            results_dict.update({img_id: result})

    return results_dict

def inference(
        fine_model,
        coarse_model,
        data_loader,
        nms_thresh,
):
    device = torch.device('cuda')
    dataset = data_loader.dataset
    predictions = predict(fine_model, coarse_model, data_loader, device)
    return evaluate(dataset=dataset, predictions=predictions, nms_thresh=nms_thresh)
