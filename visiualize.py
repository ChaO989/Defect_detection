import os
import torch
import json
from PIL import Image
from lxml import etree
import numpy as np
from draw_box_utils import draw_objs
import matplotlib.pyplot as plt


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}
def main():
    txt_path = "VOCdevkit3/VOC2007/ImageSets/Main/test.txt"
    with open(txt_path) as read:
        xml_list = [os.path.join("VOCdevkit3/VOC2007/Annotations", line.strip() + ".xml")
                    for line in read.readlines() if len(line.strip()) > 0]
    label_json_path = './pascal_voc_classes2.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)
    category_index = {str(v): str(k) for k, v in class_dict.items()}
    for i in range(len(xml_list)):
        with open(xml_list[i]) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        original_img = Image.open(os.path.join("VOCdevkit3/VOC2007/JPEGImages/", data["filename"]))
        boxes = []
        labels = []
        scores = []
        if "object" in data:
            for obj in data["object"]:
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_dict[obj["name"]])
                scores.append(1)
        boxes = np.array(boxes)
        labels = np.array(labels)
        scores = np.array(scores)
        plot_img = draw_objs(original_img,
                             boxes,
                             labels,
                             scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        # from PIL import Image
        # import matplotlib.pyplot as plt
        # 保存预测的图片结果
        plot_img.save(os.path.join("visiualize", str(i) + ".jpg"))
if __name__ == '__main__':
    main()