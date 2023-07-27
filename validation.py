"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np

import transforms
from network_files import FasterRCNN
from backbone import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import get_coco_api_from_dataset, CocoEvaluator
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from backbone import BackboneWithFPN, LastLevelMaxPool
from network_files import FasterRCNN, AnchorsGenerator
import matplotlib
import random
import numpy as np
import matplotlib.pyplot as plt


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                if iouThr == 0.9:
                    t = [8]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    # stats[0], print_list[0] = _summarize(1)
    # stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    # stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    # stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    # stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    # stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    # stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    # stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    # stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    # stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    # stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    # stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
    stats[0], print_list[0] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[1], print_list[1] = _summarize(0, iouThr=.55, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(0, iouThr=.6, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(0, iouThr=.65, maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(0, iouThr=.7, maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, iouThr=.8, maxDets=self.params.maxDets[2])
    stats[7], print_list[7] = _summarize(0, iouThr=.85, maxDets=self.params.maxDets[2])
    stats[8], print_list[8] = _summarize(0, iouThr=.9, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, iouThr=.95, maxDets=self.params.maxDets[2])


    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {v: k for k, v in class_dict.items()}

    VOC_root = parser_data.data_path
    # check voc root
    # if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
    #     raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = VOCDataSet(VOC_root, "2007", data_transform["val"], "val.txt")
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=nw,
                                                     pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = torchvision.models.efficientnet_v2_m()
    # print(backbone)
    return_layers = {"features.2": "0",
                     "features.3": "1",# stride 8 # stride 16
                     "features.8": "2"}  # stride 32
    # 提供给fpn的每个特征层channel
    in_channels_list = [48,80,1280]
    new_backbone = create_feature_extractor(backbone, return_layers)
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率
    model = FasterRCNN(backbone=backbone_with_fpn, num_classes=parser_data.num_classes + 1,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)

    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    city_name = []
    data = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))
        del stats[-1:]
        for j in range(9):
            stats.insert(0, stats[0])
        X = np.linspace(0, 1, 20)  # X轴坐标数据
        plt.figure(figsize=(4, 4))  # 定义图的大小
        plt.xlabel("Score_Threhold")  # X轴标签
        plt.ylabel("Recall")  # Y轴坐标标签
        plt.title("{}={:.2%}\nscore_threhold = 0.5".format(category_index[i + 1], stats[0]))  # 曲线图的标题
        plt.xlim(0, 1)
        plt.ylim(0, 1.03)
        plt.plot(X, stats, linewidth=2)  # 绘制曲线图
        plt.savefig("recall/{}.jpg".format(category_index[i + 1]))
        city_name.append(category_index[i + 1])
        data.append(stats[1])

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)
    fig, ax = plt.subplots_adjust(left = 0.2)
    b = ax.barh(range(len(city_name)), data, color='#6699CC')

    # 为横向水平的柱图右侧添加数据标签。
    for rect in b:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%.3f' %
                float(w), ha='left', va='center')

    # 设置Y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(city_name)))
    ax.set_yticklabels(city_name)

    # 不要X横坐标上的label标签。
    plt.xticks(())

    plt.title("mAP={:.2%}".format(data[1]), loc='center', fontsize='10',
              fontweight='bold', color='black')

    plt.show()

    # 将验证结果保存至txt文件中
    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default='6', help='number of classes')

    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='./', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='weight/resNetFpn-model-15.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when validation.')

    args = parser.parse_args()

    main(args)
