import torch
from ..utils.inference import inference
from torch import optim
from ..dataset.data_preprocess import make_data_loader
from ..fineGraind.fineGraindModule import fine
from ..coarse.coarseModule import coarse

max_epoch = 15
device = "cuda"
LR = 0.0013
MS = (7, 11, 14)

def train(
        fine_model,
        coarse_model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
):
    fine_model.train()
    coarse_model.train()
    for epoch in range(max_epoch + 1):
        for iteration, (batches, targets, _) in enumerate(data_loader):
            iteration += 1
            batches = batches.to(device)
            targets = targets.to(device)
            def closure():
                optimizer.zero_grad()
                f_loss = fine_model(batches, targets)
                c_loss = coarse_model(batches)
                loss = f_loss + c_loss
                loss.backward()
                return loss
            optimizer.step(closure)
        scheduler.step()

        torch.save(coarse_model.state_dict(), f"output/coarse_model_{epoch}.pth")
        torch.save(fine_model.state_dict(), f"output/fine_model_{epoch}.pth")

        if data_loader_val is not None:
            inference(
                fine_model,
                coarse_model,
                data_loader_val,
                nms_thresh=0.5,
            )
            fine_model.train()
            coarse_model.train()

if __name__ == '__main__':
    fine_model = fine.module()
    coarse_model = coarse.module()
    fine_model.to(device)
    coarse_model.to(device)
    optimizer = optim.Adam([{'params': fine_model.parameters()}, {'params': coarse_model.parameters()}], lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MS)
    data_loader_train = make_data_loader(
        is_train=True
    )
    data_loader_val = make_data_loader(is_train=False, is_for_period=True)
    train(
        fine_model,
        coarse_model,
        data_loader_train,
        data_loader_val,
        optimizer,
        scheduler,
        )

    data_loader_val = make_data_loader(is_train=False)
    inference(
        fine_model,
        coarse_model,
        data_loader_val,
        nms_thresh=0.5,
    )