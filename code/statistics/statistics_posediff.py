import os
import sys
from unicodedata import category
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, os.path.join(BASEPATH, '..'))
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from utils.pose import get_pose_diff, compute_IoU_with_symaxis
from utils.kinematic import pred_poses_from_kinematics
from transforms3d.euler import mat2euler
from utils.get_framelist import get_framelist


def eval(pred_poses, gt_poses, ids, result_path, sym_axis=None, output_path=None):
    os.makedirs(result_path, exist_ok=True)

    cnt = 0
    mean_tdiff = 0
    mean_rdiff = 0
    acc_3_3 = 0
    acc_5_5 = 0
    acc_3_03 = 0
    acc_5_05 = 0
    acc_10_1 = 0
    acc_20_2 = 0
    
    x = []
    rdiffs = []
    tdiffs = []
    tdiffs_detailed = []

    for idx in range(len(ids)):
        if idx == 0:
            continue
        _, tdiff, rdiff = get_pose_diff(pred_poses[idx], gt_poses[idx], sym_axis=sym_axis)
        cnt += 1
        mean_tdiff += tdiff * 1000
        mean_rdiff += rdiff
        if (tdiff <= 0.03) and (rdiff <= 3):
            acc_3_3 += 1
        if (tdiff <= 0.05) and (rdiff <= 5):
            acc_5_5 += 1
        if (tdiff <= 0.003) and (rdiff <= 3):
            acc_3_03 += 1
        if (tdiff <= 0.005) and (rdiff <= 5):
            acc_5_05 += 1
        if (tdiff <= 0.010) and (rdiff <= 10):
            acc_10_1 += 1
        if (tdiff <= 0.020) and (rdiff <= 20):
            acc_20_2 += 1
        
        x.append(idx)
        rdiffs.append(rdiff)
        tdiffs.append(tdiff * 1000)

        tdiffs_detailed.append(pred_poses[idx]["translation"].reshape(3) - gt_poses[idx]["translation"].reshape(3))

    mean_tdiff /= cnt
    mean_rdiff /= cnt
    acc_3_3 /= cnt
    acc_5_5 /= cnt
    acc_3_03 /= cnt
    acc_5_05 /= cnt
    acc_10_1 /= cnt
    acc_20_2 /= cnt
    print("mean_tdiff", mean_tdiff)
    print("mean_rdiff", mean_rdiff)
    print("acc_3_3", acc_3_3)
    print("acc_5_5", acc_5_5)
    print("acc_3_03", acc_3_03)
    print("acc_5_05", acc_5_05)
    print("acc_10_1", acc_10_1)
    print("acc_20_2", acc_20_2)

    if not output_path is None:
        wr = open(output_path, "w")
        wr.write("mean_tdiff(mm) = " + str(mean_tdiff) + "\n")
        wr.write("mean_rdiff(deg) = " + str(mean_rdiff) + "\n")
        wr.write("3deg3cm = " + str(acc_3_3) + "\n")
        wr.write("5deg5cm = " + str(acc_5_5) + "\n")
        wr.write("3deg3mm = " + str(acc_3_03) + "\n")
        wr.write("5deg5mm = " + str(acc_5_05) + "\n")
        wr.write("10deg10mm = " + str(acc_10_1) + "\n")
        wr.write("20deg20mm = " + str(acc_20_2) + "\n")
        wr.close()

    # draw curves
    plt.clf()
    plt.plot(x, rdiffs, color="red", label="rdiff")
    plt.plot(x, tdiffs, color="blue", label="tdiff")
    plt.legend()
    plt.title("diff")
    plt.xlabel("frame")
    plt.ylabel("(mm) (deg)")
    plt.savefig(os.path.join(result_path, "pose_diffs.png"))
    plt.clf()
    tdiffs_detailed = np.float32(tdiffs_detailed)
    plt.plot(x, tdiffs_detailed[:, 0], color="red", label="tdiff[0]")
    plt.plot(x, tdiffs_detailed[:, 1], color="green", label="tdiff[1]")
    plt.plot(x, tdiffs_detailed[:, 2], color="blue", label="tdiff[2]")
    plt.legend()
    plt.title("tdiff")
    plt.xlabel("frame")
    plt.ylabel("(mm)")
    plt.savefig(os.path.join(result_path, "tdiffs.png"))
