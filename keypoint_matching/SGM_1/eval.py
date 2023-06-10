#The following evaluation file is a modification of the eval.py available in Thinkmatch - https://github.com/Thinklab-SJTU/ThinkMatch/, Copyright (c) 2021 Thinklab@SJTU, ThinkMatch is licensed under Mulan PSL v2. You can use this software according to the terms and conditions of the Mulan PSL v2.



import time
from datetime import datetime
from pathlib import Path
import xlwt

from src.dataset.data_loader import GMDataset, get_dataloader
from src.evaluation_metric import *
from src.parallel import DataParallel
from src.utils.model_sl import load_model
from src.utils.data_to_cuda import data_to_cuda
from src.utils.timer import Timer

from src.utils.config import cfg
from pygmtools.benchmark import Benchmark


def eval_model(model, classes, bm, last_epoch=True, verbose=False, xls_sheet=None):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    dataloaders = []

    for cls in classes:
        image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                                  bm,
                                  cfg.EVAL.SAMPLES,
                                  cfg.PROBLEM.TEST_ALL_GRAPHS,
                                  cls,
                                  cfg.PROBLEM.TYPE)

        torch.manual_seed(cfg.RANDOM_SEED) 

        dataloader = get_dataloader(image_dataset, shuffle=True)
        dataloaders.append(dataloader)

    recalls = []
    precisions = []
    f1s = []
    coverages = []
    pred_time = []
    objs = torch.zeros(len(classes), device=device)
    cluster_acc = []
    cluster_purity = []
    cluster_ri = []

    timer = Timer()

    prediction = []

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        pred_time_list = []
        obj_total_num = torch.zeros(1, device=device)
        cluster_acc_list = []
        cluster_purity_list = []
        cluster_ri_list = []
        prediction_cls = []

        for inputs in dataloaders[i]:
            if iter_num >= cfg.EVAL.SAMPLES / inputs['batch_size']:
                break
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)

            batch_num = inputs['batch_size']

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                timer.tick()
                outputs = model(inputs)
                pred_time_list.append(torch.full((batch_num,), timer.toc() / batch_num))

            
            if cfg.PROBLEM.TYPE == '2GM':
                assert 'perm_mat' in outputs

                for b in range(outputs['perm_mat'].shape[0]):
                    perm_mat = outputs['perm_mat'][b, :outputs['ns'][0][b], :outputs['ns'][1][b]].cpu()
                    perm_mat = perm_mat.numpy()
                    eval_dict = dict()
                    id_pair = inputs['id_list'][0][b], inputs['id_list'][1][b]
                    eval_dict['ids'] = id_pair
                    eval_dict['cls'] = cls
                    eval_dict['perm_mat'] = perm_mat
                    prediction.append(eval_dict)
                    prediction_cls.append(eval_dict)

                if 'aff_mat' in outputs:
                    pred_obj_score = objective_score(outputs['perm_mat'], outputs['aff_mat'])
                    gt_obj_score = objective_score(outputs['gt_perm_mat'], outputs['aff_mat'])
                    objs[i] += torch.sum(pred_obj_score / gt_obj_score)
                    obj_total_num += batch_num
           
            else:
                raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        objs[i] = objs[i] / obj_total_num
        pred_time.append(torch.cat(pred_time_list))
       
       

    
    
    result = bm.eval(prediction, classes, verbose=True)
    for cls in classes:
            precision = result[cls]['precision']
            recall = result[cls]['recall']
            f1 = result[cls]['f1']
            coverage = result[cls]['coverage']

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            coverages.append(coverage)

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)

    if xls_sheet:
        for idx, cls in enumerate(classes):
            xls_sheet.write(0, idx+1, cls)
        xls_sheet.write(0, idx+2, 'mean')

    xls_row = 1

    
    if xls_sheet:
        xls_sheet.write(xls_row, 0, 'precision')
        xls_sheet.write(xls_row+1, 0, 'recall')
        xls_sheet.write(xls_row+2, 0, 'f1')
        xls_sheet.write(xls_row+3, 0, 'coverage')
    for idx, (cls, cls_p, cls_r, cls_f1, cls_cvg) in enumerate(zip(classes, precisions, recalls, f1s, coverages)):
        if xls_sheet:
            xls_sheet.write(xls_row, idx+1, '{:.4f}'.format(cls_p)) 
            xls_sheet.write(xls_row+1, idx+1, '{:.4f}'.format(cls_r)) 
            xls_sheet.write(xls_row+2, idx+1, '{:.4f}'.format(cls_f1)) 
            xls_sheet.write(xls_row+3, idx+1, '{:.4f}'.format(cls_cvg))
    if xls_sheet:
        xls_sheet.write(xls_row, idx+2, '{:.4f}'.format(result['mean']['precision'])) 
        xls_sheet.write(xls_row+1, idx+2, '{:.4f}'.format(result['mean']['recall']))
        xls_sheet.write(xls_row+2, idx+2, '{:.4f}'.format(result['mean']['f1'])) 
        xls_row += 4

    if not torch.any(torch.isnan(objs)):
        print('Normalized objective score')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'norm objscore')
        for idx, (cls, cls_obj) in enumerate(zip(classes, objs)):
            print('{} = {:.4f}'.format(cls, cls_obj))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, cls_obj.item()) 
        print('average objscore = {:.4f}'.format(torch.mean(objs)))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, torch.mean(objs).item())
            xls_row += 1


    print('Predict time')
    if xls_sheet: xls_sheet.write(xls_row, 0, 'time')
    for idx, (cls, cls_time) in enumerate(zip(classes, pred_time)):
        print('{} = {}'.format(cls, format_metric(cls_time)))
        if xls_sheet: xls_sheet.write(xls_row, idx + 1, torch.mean(cls_time).item())
    print('average time = {}'.format(format_metric(torch.cat(pred_time))))
    if xls_sheet:
        xls_sheet.write(xls_row, idx+2, torch.mean(torch.cat(pred_time)).item())
        xls_row += 1

    bm.rm_gt_cache(last_epoch=last_epoch)

    return torch.Tensor(recalls)


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)

    ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
    benchmark = Benchmark(name=cfg.DATASET_FULL_NAME,
                          sets='test',
                          problem=cfg.PROBLEM.TYPE,
                          obj_resize=cfg.PROBLEM.RESCALE,
                          filter=cfg.PROBLEM.FILTER,
                          **ds_dict)

    cls = None if cfg.EVAL.CLASS in ['none', 'all'] else cfg.EVAL.CLASS
    if cls is None:
        clss = benchmark.classes
    else:
        clss = [cls]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('epoch{}'.format(cfg.EVAL.EPOCH))
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)

        model_path = ''
        if cfg.EVAL.EPOCH is not None and cfg.EVAL.EPOCH > 0:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(cfg.EVAL.EPOCH))
        if len(cfg.PRETRAINED_PATH) > 0:
            model_path = cfg.PRETRAINED_PATH
        if len(model_path) > 0:
            print('Loading model parameters from {}'.format(model_path))
            load_model(model, model_path)

        pcks = eval_model(
            model, clss,
            benchmark,
            verbose=True,
            xls_sheet=ws
        )
    wb.save(str(Path(cfg.OUTPUT_PATH) / ('eval_result_' + now_time + '.xls')))
