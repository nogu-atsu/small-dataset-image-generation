import os, time
from pathlib import Path
import shutil
import numpy as np
import argparse
import chainer
from chainer import cuda
from chainer.links import VGG16Layers as VGG
from chainer.training import extensions
import chainermn

import yaml
import source.yaml_utils as yaml_utils
from gen_models.ada_generator import AdaBIGGAN, AdaSNGAN
from dis_models.patch_discriminator import Discriminator as PatchDiscriminator
from updater import Updater


def get_dataset(image_size, config):
    # return an array of image shaped (config.datasize, 3, image_size, image_size)

    if config.dataset == "dataset_name":
        # please define your dataset here if necessary
        pass

    # default dataset
    # images in {config.data_path}/{config.dataset} directory are loaded
    else:
        import cv2
        img_path = Path(f"{config.data_path}/{config.dataset}")
        img_path = list(img_path.glob("*"))[:config.datasize]
        img = []
        for i in range(config.datasize):
            img_ = cv2.imread(str(img_path[i]))[:, :, ::-1]
            h, w = img_.shape[:2]
            size = min(h, w)
            img_ = img_[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
            img.append(cv2.resize(img_, (image_size, image_size)))
        img = np.array(img).transpose(0, 3, 1, 2)
        img = img.astype("float32") / 127.5 - 1

    print("number of data", len(img))

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/default.yml")
    # parser.add_argument("--resume", "-r", type=str, default="")
    parser.add_argument("--communicator", type=str, default="hierarchical")
    parser.add_argument("--suffix", type=int, default=0)

    args = parser.parse_args()
    now = int(time.time()) * 10 + args.suffix
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    os.makedirs(f"{config.save_path}{now}", exist_ok=True)
    shutil.copy(args.config_path, f"{config.save_path}{now}/config{now}.yml")
    shutil.copy("train.py", f"{config.save_path}{now}/train.py")
    print("snapshot->", now)

    # image size
    config.image_size = config.image_sizes[config.gan_type]
    image_size = config.image_size

    if config.gan_type == "BIGGAN":
        try:
            comm = chainermn.create_communicator(args.communicator)
        except:
            comm = None
    else:
        comm = None

    device = args.gpu if comm is None else comm.intra_rank
    cuda.get_device(device).use()

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu)
        xp = cuda.cupy
    else:
        xp = np

    np.random.seed(1234)

    if config.perceptual:
        vgg = VGG().to_gpu()
    else:
        vgg = None

    layers = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2",
              "conv4_3"]


    img = xp.array(get_dataset(image_size, config))

    if comm is None or comm.rank == 0:
        perm_dataset = np.arange(len(img))
    else:
        perm_dataset = None

    if comm is not None:
        perm_dataset = chainermn.scatter_dataset(perm_dataset, comm, shuffle=True)

    batchsize = min(img.shape[0], config.batchsize[config.gan_type])
    perm_iter = chainer.iterators.SerialIterator(perm_dataset, batch_size=batchsize)

    ims = []
    datasize = len(img)

    target = img

    # Model
    if config.gan_type == "BIGGAN":
        gen = AdaBIGGAN(config, datasize, comm=comm)
    elif config.gan_type == "SNGAN":
        gen = AdaSNGAN(config, datasize, comm=comm)

    if not config.random:  # load pre-trained generator model
        chainer.serializers.load_npz(config.snapshot[config.gan_type], gen.gen)
    gen.to_gpu(device)
    gen.gen.to_gpu(device)

    if config.l_patch_dis > 0:
        dis = PatchDiscriminator(comm=comm)
        dis.to_gpu(device)
        opt_dis = dis.optimizer
        opts = {"opt_gen": gen.optimizer, "opt_dis": opt_dis}
    else:
        dis = None
        opt_dis = None
        opts = {"opt_gen": gen.optimizer}

    models = {"gen": gen, "dis": dis}

    kwargs = {"gen": gen, "dis": dis, "vgg": vgg, "target": target, "layers": layers, "optimizer": opts,
              "iterator": perm_iter, "device": device, "config": config}
    updater = Updater(**kwargs)

    trainer = chainer.training.Trainer(updater, (config.iteration, 'iteration'), out=f"{config.save_path}{now}")

    if comm is None or comm.rank == 0:
        report_keys = ['epoch', 'iteration', 'loss_gen', 'loss_dis']

        trainer.extend(extensions.snapshot_object(gen, "gen" + str(now) + "_{.updater.iteration}.h5"),
                       trigger=(config.snapshot_interval, "iteration"))
        trainer.extend(extensions.snapshot_object(gen.gen, "gen_gen" + str(now) + "_{.updater.iteration}.h5"),
                       trigger=(config.snapshot_interval, "iteration"))
        if dis is not None:
            trainer.extend(extensions.snapshot_object(dis, "dis" + str(now) + "_{.updater.iteration}.h5"),
                           trigger=(config.snapshot_interval, "iteration"))

        trainer.extend(extensions.LogReport(trigger=(config.display_interval, 'iteration')))
        trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))
        # evaluation
        trainer.extend(models["gen"].evaluation(f"{config.save_path}{now}"),
                       trigger=(config.evaluation_interval, 'iteration'))

    trainer.run()
