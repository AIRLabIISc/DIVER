import numpy as np
import cv2 as cv
from collections import OrderedDict
import os

def obtain_BGR_img(img_array):
    """
    Stack BGR channels from iterable of [array, 'B'/'G'/'R'] pairs

    """
    m = {ch: arr for (arr, ch) in img_array}
    return np.dstack([m['B'], m['G'], m['R']]).astype(np.uint8)

def sorting_BGR_channel(img):
    B, G, R = cv.split(img)
    channel = {"B": np.mean(B), "G": np.mean(G), "R": np.mean(R)}
    channel_ordered = dict(sorted(channel.items(), key=lambda x: x[1], reverse=True))

    label = ["Cmax", "Cint", "Cmin"]
    chanel_sorted = {}
    for i, j in zip(range(len(label)), channel_ordered.keys()):
        if j == "B":
            chanel_sorted[label[i]] = [B, j]
        elif j == "G":
            chanel_sorted[label[i]] = [G, j]
        else:
            chanel_sorted[label[i]] = [R, j]
    return chanel_sorted

def contrast_enhancement_filter(image):
    """
    Vectorized dual-intensity stretching 

    """
    img = image.astype(np.float32)
    LS = np.zeros_like(img, dtype=np.float32)
    US = np.zeros_like(img, dtype=np.float32)

    for c in range(3):
        ch = img[..., c]
        minC = float(ch.min()); maxC = float(ch.max())
        meanC = float(ch.mean()); medC = float(np.median(ch))
        avg = 0.5 * (meanC + medC)

        denom_ls = max(avg - minC, 1e-6)
        denom_us = max(maxC - avg, 1e-6)

        ls = (ch - minC) * ((255.0 - minC) / denom_ls) + minC
        us = (ch - avg) * (255.0 / denom_us)

        LS[..., c] = np.where(ch < avg, ls, 255.0)
        US[..., c] = np.where(ch < avg, 0.0, us)

    return LS.clip(0, 255).astype(np.uint8), US.clip(0, 255).astype(np.uint8)

def image_fusion(img1, img2, use_mertens=False):
    """
    Fuse the two intensity images
    
    """
    if use_mertens:
        try:
            merge = cv.createMergeMertens(contrast_weight=1.0, saturation_weight=0.5, exposure_weight=1.0)
            f = merge.process([img1.astype(np.float32)/255.0, img2.astype(np.float32)/255.0])
            return (f * 255.0).clip(0, 255).astype(np.uint8)
        except Exception:
            pass  # fall back to average

    b1, g1, r1 = cv.split(img1)
    b2, g2, r2 = cv.split(img2)
    fused_img = np.empty_like(img1, dtype=np.uint8)
    fused_img[..., 0] = ((b1.astype(np.uint16) + b2.astype(np.uint16)) // 2).astype(np.uint8)
    fused_img[..., 1] = ((g1.astype(np.uint16) + g2.astype(np.uint16)) // 2).astype(np.uint8)
    fused_img[..., 2] = ((r1.astype(np.uint16) + r2.astype(np.uint16)) // 2).astype(np.uint8)
    return fused_img



def suppress_hue_artifacts(bgr,
                           hue_ranges=((70, 100),),   # cyan/aqua in OpenCV HSV (0..179)
                           s_min=160, v_min=180,      # selected HSSF experiment thresholds
                           strength=0.3,              # 0..1 pull chroma toward gray
                           method='lab'):
    """
    Mask selected hue ranges and reduce their chroma (a/b in Lab) or saturation (HSV)
    
    """
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    mask = None
    for (hL, hU) in hue_ranges:
        m = cv.inRange(hsv, np.array([hL, s_min, v_min], np.uint8),
                            np.array([hU, 255, 255],   np.uint8))
        mask = m if mask is None else cv.bitwise_or(mask, m)

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=1)

    if method == 'hsv':
        h, s, v = cv.split(hsv)
        s = s.astype(np.float32)
        s[mask > 0] = (s[mask > 0] * (1.0 - strength)).clip(0, 255)
        out = cv.merge([h, s.astype(np.uint8), v])
        return cv.cvtColor(out, cv.COLOR_HSV2BGR)
    else:
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        L, A, B = cv.split(lab)
        A = A.astype(np.float32); B = B.astype(np.float32)
        A[mask > 0] = 128.0 + (A[mask > 0] - 128.0) * (1.0 - strength)
        B[mask > 0] = 128.0 + (B[mask > 0] - 128.0) * (1.0 - strength)
        out = cv.merge([L, A.astype(np.uint8), B.astype(np.uint8)])
        return cv.cvtColor(out, cv.COLOR_LAB2BGR)

######## FINAL PIPELINE ###########

def AOCM(img, return_all=True, use_mertens=False, save_path=None, stem="output", hssf_strength=0.3):
    steps_inter = OrderedDict()
    img1, img2 = contrast_enhancement_filter(img)
    fused_img = image_fusion(img1, img2, use_mertens=use_mertens)
    steps_inter['CEF'] = fused_img
    hue_suppressed_img = suppress_hue_artifacts(
        fused_img, hue_ranges=((70, 100),), s_min=160, v_min=180,
        strength=hssf_strength, method='lab'
    )
    steps_inter['hue_suppressed'] = hue_suppressed_img
    AOCM_img = hue_suppressed_img
    steps_inter['AOCM_algorithm_output'] = AOCM_img
    if not return_all and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{stem}_AOCM.png")
        cv.imwrite(save_file, AOCM_img)

    return steps_inter if return_all else AOCM_img


def save_pipeline_steps(steps_dict, out_dir, stem):
    """
    Save all BGR uint8 images from steps_dict to out_dir with `<stem>_<stage>.png` names.
    """
    os.makedirs(out_dir, exist_ok=True)
    for stage_name, stage_img in steps_dict.items():
        fn = f"{stem}_{stage_name}.png"
        cv.imwrite(os.path.join(out_dir, fn), stage_img)
