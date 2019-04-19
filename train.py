import os, sys, time
from pathlib import Path
import shutil
import numpy as np
import argparse
import chainer
from chainer import cuda
from chainer.links import VGG16Layers as VGG
from chainer.training import extensions
import chainermn
from PIL import Image

sys.path.append("evaluations/")
import yaml
import source.yaml_utils as yaml_utils
from gen_models.ada_biggan import AdaBIGGAN, AdaSNGAN
from dis_models.patch_discriminator import Discriminator as PatchDiscriminator
from updater import Updater


def get_dataset(image_size, config):
    # return an array of image shaped (N, 3, height, width)

    # celeba 128
    if config.dataset == "face":
        assert image_size == 128, "invalid size"
        if config.datasize < 25:
            img_name = [f"{i + 1:0>6}.jpg" for i in [0, 1, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15][:config.datasize]]
        else:
            img_name = [f"{i + 1:0>6}.jpg" for i in range(config.datasize)]
        img = []
        for name in img_name:
            img.append(np.array(Image.open(f"{config.data_path}CelebA/celebA/celeba128/" + name)))
        img = np.array(img).transpose(0, 3, 1, 2)
        img = img.astype("float32") / 127.5 - 1
        print("number of data", len(img))

    # oxford 102
    if config.dataset[:6] == "flower":
        import cv2

        if config.dataset == "flower":
            flower = Path(f"{config.data_path}102flowers/")
            flower = list(flower.glob("*.jpg"))[:config.datasize]
        else:
            cat = int(config.dataset.split("_")[-1])
            import scipy.io as io

            mat = io.loadmat(f"{config.data_path}102flowers/imagelabels.mat")["labels"][0]
            ids = np.where(mat == cat)[0] + 1
            flower = [f"{config.data_path}102flowers/image_{i:0>5}.jpg" for i in ids[:config.datasize]]
            assert len(flower) == config.datasize, "datasize is not correct"
        img = []
        for i in range(config.datasize):
            img_ = cv2.imread(str(flower[i]))[:, :, ::-1]
            h, w = img_.shape[:2]
            size = min(h, w)
            img_ = img_[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
            img.append(cv2.resize(img_, (image_size, image_size)))
        img = np.array(img).transpose(0, 3, 1, 2)
        img = img.astype("float32") / 127.5 - 1
        print("number of data", len(img))

    if config.dataset == "FFHQ":
        import cv2
        if config.datasize <= 50:
            ffhq = Path(f"{config.data_path}FFHQ/")
            ffhq = list(ffhq.glob("*.png"))[:config.datasize]
            img = []
            for i in range(config.datasize):
                img_ = cv2.imread(str(ffhq[i]))[:, :, ::-1]
                h, w = img_.shape[:2]
                size = min(h, w)
                img_ = img_[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
                img.append(cv2.resize(img_, (image_size, image_size)))
            img = np.array(img).transpose(0, 3, 1, 2)
            img = img.astype("float32") / 127.5 - 1
            print("number of data", len(img))
        else:
            ffhq = Path(f"{config.data_path}ffhq500/")
            ffhq = list(ffhq.glob("*.png"))[:config.datasize]
            img = []
            for i in range(config.datasize):
                img_ = cv2.imread(str(ffhq[i]))[:, :, ::-1]
                h, w = img_.shape[:2]
                size = min(h, w)
                img_ = img_[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
                img.append(cv2.resize(img_, (image_size, image_size)))
            img = np.array(img).transpose(0, 3, 1, 2)
            img = img.astype("float32") / 127.5 - 1
            print("number of data", len(img))

    elif config.dataset == "Anime":
        import cv2
        if config.datasize <= 50:
            ffhq = Path(f"{config.data_path}anime/aligned56")
            ffhq = list(ffhq.glob("*.jpg"))[:config.datasize]
            img = []
            for i in range(config.datasize):
                img_ = cv2.imread(str(ffhq[i]))[:, :, ::-1]
                h, w = img_.shape[:2]
                size = min(h, w)
                img_ = img_[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
                img.append(cv2.resize(img_, (image_size, image_size)))
            img = np.array(img).transpose(0, 3, 1, 2)
            img = img.astype("float32") / 127.5 - 1
            print("number of data", len(img))
        else:
            ffhq = Path(f"{config.data_path}anime/waifu_128")
            ffhq = list(ffhq.glob("*.jpg"))[:config.datasize]
            img = []
            for i in range(config.datasize):
                img_ = cv2.imread(str(ffhq[i]))[:, :, ::-1]
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
    parser.add_argument("--results_dir", type=str, default="./results/gans")
    parser.add_argument("--row", type=int, default=5)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--classes", default=None)
    parser.add_argument("--resume", "-r", type=str, default="")
    parser.add_argument("--communicator", type=str, default="hierarchical")
    parser.add_argument("--suffix", type=int, default=0)

    args = parser.parse_args()
    now = int(time.time()) * 10 + args.suffix
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    os.makedirs(f"{config.save_path}{now}", exist_ok=True)
    shutil.copy(args.config_path, f"{config.save_path}{now}/config{now}.yml")
    shutil.copy("fewshot_generation_mn.py", f"{config.save_path}{now}/fewshot_generation_mn.py")
    print("snapshot->", now)

    # image size
    config.image_size = {"SNGAN": 128, "BIGGAN": 256}[config.gan_type]

    if config.gan_type == "BIGGAN":
        comm = chainermn.create_communicator(args.communicator)
    else:
        comm = None
    # args = Args()

    device = args.gpu if comm is None else comm.intra_rank  # プロセスのランクでGPUデバイスを割り振る
    chainer.cuda.get_device(device).use()

    xp = cuda.cupy

    np.random.seed(1234)

    if config.perceptual:
        vgg = VGG().to_gpu()
    else:
        vgg = None

    layers = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2",
              "conv4_3"]

    image_size = config.image_size

    img = xp.array(get_dataset(image_size, config))

    if comm is None or comm.rank == 0:  # ランク0のプロセスでのみデータセットを作る
        perm_dataset = np.arange(len(img))
    else:
        perm_dataset = None

    if comm is not None:
        perm_dataset = chainermn.scatter_dataset(perm_dataset, comm, shuffle=True)

    batchsize = min(img.shape[0], config.batchsize[config.gan_type])
    perm_iter = chainer.iterators.SerialIterator(perm_dataset, batch_size=batchsize)

    ims = []
    datasize = len(img)
    print(datasize)

    target = img

    # Model
    if config.gan_type == "BIGGAN":
        gen = AdaBIGGAN(config, datasize, comm=comm)
    elif config.gan_type == "SNGAN":
        n_classes = config.n_classes if hasattr(config, 'n_classes') else 1000
        gen = AdaSNGAN(config, datasize, n_classes=n_classes, comm=comm)

    out = args.results_dir
    if not config.random:
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
