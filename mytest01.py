from __future__ import print_function
import json
import sklearn.metrics as metrics
import tqdm
from setuptools.sandbox import save_path
from torch.utils.data import DataLoader
from TSegFormer2.main import seg2colors
from data import Teeth
from model import TSegFormer
from util import *

file_path=



def inference(args, io):
    f = open(args.file_path, 'r')
    device = torch.device("cuda" if args.cuda else "cpu")
    model = TSegFormer(args, 33).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    teeth_dict = json.load(f)
    feature = np.array(teeth_dict['feature'][:args.num_points], dtype=np.float32)
    xyz = feature[:, 0:3]
    feature = torch.Tensor(feature).type(torch.FloatTensor).to(device).unsqueeze(0).permute(0, 2, 1)
    seg_gold = np.array(teeth_dict['label'][:args.num_points], dtype=np.int64)
    category = torch.Tensor(teeth_dict['category']).type(torch.FloatTensor).to(device).unsqueeze(0)
    seg_pred = model(feature, category)[0].permute(0, 2, 1)
    seg_pred = seg_pred.max(dim=2)[1].reshape(-1).cpu().numpy()
    if category[0, 0] == 1:
        seg_pred[seg_pred > 0] += 16
    acc = metrics.accuracy_score(seg_gold, seg_pred)
    class_averaged_recall = metrics.balanced_accuracy_score(seg_gold, seg_pred)
    iou = calculate_shape_IoU(np.expand_dims(seg_pred, 0), np.expand_dims(seg_gold, 0), None)[0]

    print("Inference Accuracy             :", acc)
    print("Inference Class-averaged Recall:", class_averaged_recall)
    print("part-averaged IoU              :", iou)
    print("set of teeth in gold           :", set(seg_gold))
    print("set of teeth in gold           :", set(seg_pred.flatten()))
    generate_obj(xyz, seg_pred, "objs/pred.obj")
    generate_obj(xyz, seg_gold, "objs/gold.obj")


def generate_obj(xyz, seg, path):
    print(xyz.shape)
    print(seg.shape)
    with open(path, "w") as f:
        for idx in range(xyz.shape[0]):
            f.write(
                "v %.6f %.6f %.6f %.4f %.4f %.4f\n" % ((xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]) + seg2colors[seg[idx]]))



