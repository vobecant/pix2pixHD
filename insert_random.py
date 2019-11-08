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
import random


def insert2label(mask_bin, orig_label_map, pedestrian_id=24):
    mask_npy = np.asarray(mask_bin)
    mask_pedId = mask * pedestrian_id
    mask2insert = Image.fromarray(mask_pedId)
    mask_w, mask_h = masks2insert.size

    im_h, im_w, _ = orig_label_map.shape
    max_w = im_w - mask_w
    max_h = im_h - mask_h
    x, y = random.randrange(0, max_w), random.randrange(0, max_h)

    orig_label_map = Image.fromarray(orig_label_map)
    orig_label_map = orig_label_map.paste(mask2insert, (x, y), mask_bin)
    orig_label_map = np.asarray(orig_label_map)

    bb_inserted = (x, y, mask_w, mask_h)

    return orig_label_map, bb_inserted


def fit_to_image(x, y, side, im_w, im_h):
    '''
    params:
        (x,y): top-left corner of the crop
        side: crop size of a square box in px
        im_w, im_h: image width and height
    output:
        modified values (x,y) such that the whole box is inside an image
    '''

    # move the top-left corner so that it is valid first
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    # coordinates of the bottom right corned
    x2, y2 = x + side, y + side
    xsize = abs(x - x2)
    ysize = abs(y - y2)

    max_x, max_y = im_w - 1, im_h - 1

    if x2 > max_x:
        # delta: how much do we need to shift the top-left x-coordinate
        delta_x = x2 - max_x
        x = max([0, x - delta_x])
        x2 = min([max_x, x2])
        xsize = abs(x - x2)
    if y2 > max_y:
        delta_y = y2 - max_y
        y = max([0, y - delta_y])
        y2 = min([max_y, y2])
        ysize = abs(y - y2)

    if (x < 0) or (y < 0):
        raise Exception(
            "x and y cannot be negative! (x={:d}, x2={}, dx={}, y={:d}, y2={:d}, dy={}; "
            "image: {}x{})".format(x, x2, delta_x, y, y2, delta_y, im_h, im_w))

    orig_side = side
    side = min([xsize, ysize])
    if orig_side != side:
        pass
        # print('Original size {}, current size: {}.'.format(orig_side, side), file=sys.stderr)

    # return modified values of the top-left corner
    return x, y, side


if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    crop_size = 256
    max_shift = 64
    save_dir = '/home/vobecant/datasets/pix2pixhd/crops'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
        mask = Image.open(mask_file).convert("L")

        label_map_w_inserted, bb_inserted = insert2label(mask, data['label'])
        data['label'] = label_map_w_inserted
        lbl_h, lbl_w, _ = label_map_w_inserted.shape

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

        synthesized_image = visuals['synthesized_image']
        im_h, im_w, _ = synthesized_image.shape
        assert lbl_h == im_h and lbl_w == im_w

        # create parameters of the crop, randomly shift center of the crop
        crop_size_half = crop_size // 2
        x, y, mask_w, mask_w = bb_inserted
        shift_x = random.randrange(-max_shift, max_shift)
        shift_y = random.randrange(-max_shift, max_shift)
        x += shift_x
        y += shift_y
        x, y, side = fit_to_image(x, y, crop_size, im_w=im_w, im_h=im_h)
        crop_params = (x, y, x + side, y + side)  # (left,upper,right,lower)
        fname = os.path.join(save_dir, '{}.png'.format(n_done))
        crop = Image.fromarray(synthesized_image).crop(crop_params)
        crop.save(fname)

        dataset_idx += 1
        if dataset_idx == dataset_size:
            dataset_idx = 0
        n_done += 1

    webpage.save()
