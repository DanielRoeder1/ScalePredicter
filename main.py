#from model import ResNet
from utils import create_data_loaders, AverageMeter
from utils_guide import init_lr_scheduler, init_optim, load_config
import time


def train(epoch):
    average_meter = AverageMeter()
    global iters
    end = time.time()
    for batch_idx, (input, depth, scale_factor) in enumerate(train_loader):
        if epoch >= config.test_epoch and iters % config.test_iters == 0:
            test(epoch,batch_idx)

        model.train() # switch to train mode
        optimizer.zero_grad()

        input, scale_factor  = input.cuda(), scale_factor.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()
        
        iters+=1

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


if __name__ == '__main__':
    #model = ResNet(layers = 18)
    #model.cuda()

    config = load_config()

    optimizer = init_optim(config, net)
    lr_scheduler = init_lr_scheduler(config, optimizer)

    trainloader, val_loader = create_data_loaders()

    for epoch in range(config.start_epoch, config.nepoch):
        train(epoch)
        lr_scheduler.step()
