"""
video_swapper_no_insight.py
Pipeline: ONNX SCRFD (det_10g.onnx) -> 2d106 landmarks -> ArcFace ONNX -> reswapper ONNX
Tanpa package 'insightface'.
Pastikan models ada di ./models/  :
 - models/det_10g.onnx
 - models/2d106det.onnx
 - models/w600k_r50.onnx
 - models/reswapper_256-1567500_originalInswapperClassCompatible.onnx
 - emap.npy (sudah ada di project)
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from math import ceil
from tqdm import tqdm

# -------- util : load onnx runtime sessions ----------
def make_session(path, providers=None):
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(path, providers=providers)

# --------- det/landmark/embed helper (basic) ----------
class ONNXFaceHelper:
    def __init__(self, model_dir):
        # paths (sesuaikan nama file yang kamu extract)
        self.det_path = os.path.join(model_dir, "det_10g.onnx")         # SCRFD detection
        self.lmk_path = os.path.join(model_dir, "2d106det.onnx")       # 106 landmarks
        self.arcface_path = os.path.join(model_dir, "w600k_r50.onnx")  # ArcFace embedding
        # create sessions
        self.det_sess = make_session(self.det_path)
        self.lmk_sess = make_session(self.lmk_path)
        self.arc_sess = make_session(self.arcface_path)
        # emap mapping used by ReSwapper (project embedding)
        self.emap = np.load("emap.npy") if os.path.exists("emap.npy") else None

    def detect(self, img, input_size=640, conf_th=0.4):
        # preprocess to square resize
        h0, w0 = img.shape[:2]
        scale = input_size / max(h0, w0)
        im = cv2.resize(img, (int(w0*scale), int(h0*scale)))
        # pad to input_size
        pad = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        pad[:im.shape[0], :im.shape[1]] = im
        blob = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = blob.transpose(2,0,1)[None,:,:,:]  # 1x3xHxW
        # run det
        input_name = self.det_sess.get_inputs()[0].name
        outs = self.det_sess.run(None, {input_name: blob})
        # NOTE: SCRFD onnx outputs differ by build; we assume outputs contain boxes/scores
        # For simplicity use an heuristic: parse first output if shape (...,6) as [x1,y1,x2,y2,score, ...]
        detections = []
        for out in outs:
            arr = np.array(out)
            if arr.ndim == 3 and arr.shape[-1] >= 5:
                # reshape to (N,6)
                cand = arr.reshape(-1, arr.shape[-1])
                for row in cand:
                    score = float(row[4])
                    if score >= conf_th:
                        x1 = float(row[0]) / (scale)   # mapped back to original?
                        y1 = float(row[1]) / (scale)
                        x2 = float(row[2]) / (scale)
                        y2 = float(row[3]) / (scale)
                        detections.append([x1,y1,x2,y2,score])
        # fallback: return center crop if nothing found
        if len(detections)==0:
            return []
        # sort by score
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        return detections

    def get_landmarks(self, img, bbox, input_size=192):
        # crop bbox and run 2d106 model
        x1,y1,x2,y2,_ = bbox
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        crop = img[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            return None
        h,w = crop.shape[:2]
        blob = cv2.resize(crop, (input_size, input_size))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = blob.transpose(2,0,1)[None,:,:,:]
        input_name = self.lmk_sess.get_inputs()[0].name
        outs = self.lmk_sess.run(None, {input_name: blob})
        # assume landmarks returned as (1,212) -> 106x2
        # map back to original bbox coordinates
        lm = None
        for out in outs:
            arr = np.array(out).reshape(-1)
            if arr.size >= 212:
                arr = arr[:212]
                lm = arr.reshape(-1,2)
                # scale lm back to original crop coordinates
                lm[:,0] = lm[:,0] * (x2-x1)/input_size + x1
                lm[:,1] = lm[:,1] * (y2-y1)/input_size + y1
                return lm
        return None

    def get_embedding(self, aligned_face_rgb):
        # aligned face expected RGB, HxWx3, 112x112 or 112 multiples depending model
        img = cv2.resize(aligned_face_rgb, (112,112))
        blob = img.astype(np.float32).transpose(2,0,1)[None,:,:,:]
        input_name = self.arc_sess.get_inputs()[0].name
        outs = self.arc_sess.run(None, {input_name: blob})
        emb = None
        for out in outs:
            arr = np.array(out).reshape(-1)
            if arr.size >= 512:
                emb = arr[:512]
                break
        if emb is None:
            return None
        # apply emap if available (ReSwapper expects projected latent)
        if self.emap is not None:
            latent = np.dot(emb.reshape(1,-1), self.emap)
            latent = latent / np.linalg.norm(latent)
            return latent.astype(np.float32)
        # otherwise normalize and return
        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)

# ---------- helper: align crop using 5-point ArcFace standard ----------
# we reuse face_align.norm_crop2 from your repo if present
try:
    import face_align
    have_face_align = True
except Exception:
    have_face_align = False

def align_face_using_landmarks(img, lm, image_size=256):
    # if face_align available and lm contains 5 points (eyes,nose,mouth corners)
    if have_face_align and lm.shape[0] >=5:
        # take first 5 points according to assumed order; user may need to adapt
        pts5 = lm[:5]
        aligned, M = face_align.norm_crop2(img, pts5, image_size)
        return aligned, M
    # fallback: center-crop
    h,w = img.shape[:2]
    s = min(h,w)
    cx,cy = w//2, h//2
    x1,y1 = cx-s//2, cy-s//2
    crop = img[y1:y1+s, x1:x1+s]
    M = np.eye(2,3, dtype=np.float32)
    aligned = cv2.resize(crop, (image_size, image_size))
    return aligned, M

# --------------- Main video swap function ----------------
def swap_video(input_video, source_image, output_video, model_path, model_dir):
    # load reswapper onnx
    res_sess = make_session(model_path)
    tgt_name = res_sess.get_inputs()[0].name
    src_name = res_sess.get_inputs()[1].name
    out_name = res_sess.get_outputs()[0].name

    helper = ONNXFaceHelper(model_dir)

    # get source embedding
    src_bgr = cv2.imread(source_image)
    if src_bgr is None:
        raise ValueError("source image not found or unreadable")
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    # detect face on source using same detector to get landmarks and aligned face
    dets = helper.detect(src_bgr, input_size=640)
    if len(dets)==0:
        raise ValueError("no face detected in source image")
    lm = helper.get_landmarks(src_bgr, dets[0], input_size=192)
    aligned_src, _ = align_face_using_landmarks(src_rgb, lm, image_size=256)
    src_latent = helper.get_embedding(aligned_src)  # shape (1,512) or (512,)
    if src_latent is None:
        raise ValueError("failed to compute source latent")

    # normalize shapes for ONNX: ensure shape (1,512)
    if src_latent.ndim == 1:
        src_latent = src_latent.reshape(1,-1)

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (w,h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=frame_count, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = helper.detect(frame, input_size=640)
        if len(dets)==0:
            out.write(frame)
            pbar.update(1)
            continue

        bbox = dets[0]
        lm = helper.get_landmarks(frame, bbox, input_size=192)
        if lm is None:
            out.write(frame)
            pbar.update(1)
            continue

        aligned_tgt, M = align_face_using_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), lm, image_size=256)
        # prepare blob as ReSwapper expects (Image.getBlob uses BGR->blob; here we convert aligned_tgt RGB->BGR)
        aligned_tgt_bgr = cv2.cvtColor(aligned_tgt, cv2.COLOR_RGB2BGR)
        # create blob same as Image.getBlob: 1/255 normalization and swapRB True
        blob = cv2.dnn.blobFromImage(aligned_tgt_bgr, 1.0/255.0, (256,256), (0,0,0), swapRB=True)
        # ONNX expects float inputs; ensure dtype float32
        blob = blob.astype(np.float32)
        # run reswapper
        res = res_sess.run([out_name], {tgt_name: blob, src_name: src_latent})
        out_img = np.array(res[0])[0]  # 3xHxW
        # postprocess like Image.postprocess_face: tensor->uint8 BGR
        out_img = (out_img.transpose(1,2,0) * 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)  # ensure BGR
        # blend back into original frame using mask method (simple alpha blend here)
        # Use affine inverse M to warp swapped face back
        M_inv = cv2.invertAffineTransform(M)
        warped = cv2.warpAffine(out_img, M_inv, (w,h), borderValue=0)
        # create mask from warped by convert to grayscale and threshold
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (15,15), 0)
        maskf = (mask/255.0)[:,:,None]
        # blend
        blended = (warped.astype(np.float32) * maskf + frame.astype(np.float32)*(1-maskf)).astype(np.uint8)
        out.write(blended)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print("Saved:", output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--source_image", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--model_dir", default="./models", help="folder path with det_10g.onnx, 2d106det.onnx, w600k_r50.onnx")
    parser.add_argument("--reswapper", required=True, help="reswapper onnx path (e.g. reswapper_256-... .onnx)")
    args = parser.parse_args()

    swap_video(args.input_video, args.source_image, args.output_video, args.reswapper, args.model_dir)
