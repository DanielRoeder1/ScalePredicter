
from model import ResNet
from utils import create_data_loaders, AverageMeter, load_config, save_state, Result
from utils_guide import init_lr_scheduler, init_optim
import time
import torch
import csv



def train(epoch):
    average_meter = AverageMeter()
    global iters
    end = time.time()

    if config.only_test_epoch:
        test()

    for batch_idx, (input, depth, scale_factor) in enumerate(train_loader):
        if epoch >= config.test_epoch and iters % config.test_iters == 0 and not config.only_test_epoch:
            test()

        model.train() # switch to train mode
        optimizer.zero_grad()

        input, scale_factor  = input.cuda(), scale_factor.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, scale_factor)
        
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result(pred,scale_factor)
        result.huber= loss.cpu().item()
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()
        
        iters+=1

        
        if (batch_idx + 1) % config.print_freq == 0:
            avg_huber, avg_rmse, avg_mae, avg_gpu, avg_data = average_meter.get_avg()
            print('=> output: {}'.format("checkpoints"))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({avg_data:.3f}) '
                  't_GPU={gpu_time:.3f}({avg_gpu:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({avg_rmse:.2f}) '
                  'MAE={result.mae:.2f}({avg_mae:.2f}) '
                  'Huber={result.huber:.2f}({avg_huber:.2f}) '.format(
                  epoch, batch_idx+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, avg_gpu = avg_gpu, avg_data= avg_data, 
                  avg_huber = avg_huber, avg_rmse = avg_rmse, avg_mae = avg_mae))

    
    #with open(train_csv, 'a') as csvfile:
    #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #    writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
    #        'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
    #        'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def test():
    print("=> Running test:")
    global best_metric
    average_meter = AverageMeter()
    model.eval()
    for batch_idx, (input, depth, scale_factor) in enumerate(val_loader):
        input, scale_factor = input.cuda(), scale_factor.cuda()
        data_time = 0 
        end = time.time()
        with torch.no_grad():
            pred = model(input)
            loss = criterion(pred, scale_factor)
        gpu_time = time.time() - end

        result = Result(pred,scale_factor)
        result.huber= loss.cpu().item()
        average_meter.update(result, gpu_time, data_time, input.size(0))
        if batch_idx % config.print_freq == 0:
          print(f"Batch: {batch_idx}, Loss: {loss.cpu().numpy()}")
    
    avg_huber, avg_rmse, avg_mae, _ , _ = average_meter.get_avg()
    print(f"Avg Loss: Huber: {avg_huber}, RMSE {avg_rmse}, MAE {avg_mae}")
    if avg_huber < best_metric:
        best_metric = avg_huber
        save_state(config, model)
        print('Best Result: {:.4f}\n'.format(best_metric))


if __name__ == '__main__':
    model = ResNet(layers = 18)
    model.cuda()
    print("=> Model created")

    config = load_config("configs/ScalePredicter.yaml")

    optimizer = init_optim(config, model)
    lr_scheduler = init_lr_scheduler(config, optimizer)
    criterion = torch.nn.HuberLoss()

    train_loader, val_loader = create_data_loaders(config)

    iters = 0 
    best_metric = 100

    for epoch in range(config.start_epoch, config.nepoch):
        train(epoch)
        lr_scheduler.step()
