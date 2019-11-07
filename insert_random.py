import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np
from PIL import Image


def insert2label(mask_bin, orig_label_map, pedestrian_id=-1):
    mask_npy = np.asarray(mask_bin)
    mask_pedId = mask * pedestrian_id
    mask2insert = Image.fromarray(mask_pedId)

    im_h, im_w, _ = orig_label_map.shape
    x, y = randint(0, im_w), randint(0, im_w)

    orig_label_map = Image.fromarray(orig_label_map)
    orig_label_map = orig_label_map.paste(mask2insert, (x, y), mask_bin)
    orig_label_map = np.asarray(orig_label_map)

    return orig_label_map


if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    masks2insert = np.load('')['mask_paths']
    n_masks = len(masks2insert)

    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)

        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx

    dataset_size = len(dataset)

    n_done = 0
    dataset_idx = 0
    while n_done < n_masks:
        data = dataset.__getitem__(dataset_idx)
        mask_file = masks2insert[n_done]
        mask = Image.open(mask_file)

        label_map_w_inserted = insert2label(mask, data['label'])
        data['label'] = label_map_w_inserted

        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst'] = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst'] = data['inst'].uint8()
        if opt.export_onnx:
            print("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                              opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:
            generated = model.inference(data['label'], data['inst'], data['image'])

        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                               ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

        dataset_idx += 1
        if dataset_idx == dataset_size:
            dataset_idx = 0
        n_done += 1

    webpage.save()
