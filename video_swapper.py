import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
import Image
import face_align

# ====================
# LOAD MODEL
# ====================
onnx_path = "reswapper_256-1567500_originalInswapperClassCompatible.onnx"
session = ort.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# input names
input_target_name = session.get_inputs()[0].name
input_source_name = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name

# ====================
# FACE ANALYSIS (InsightFace)
# ====================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ====================
# SOURCE LATENT
# ====================
def get_latent(path):
    img = cv2.imread(path)
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in source")
    return Image.getLatent(faces[0])  # (1,512)

# ====================
# TARGET PREP
# ====================
def prepare_target(frame):
    faces = face_app.get(frame)
    if len(faces) == 0:
        return None, None, None

    face = faces[0]
    aligned, M = face_align.norm_crop2(frame, face.kps, 256)
    blob = Image.getBlob(aligned, (256, 256))
    return blob, M, face

# ====================
# VIDEO SWAP LOOP
# ====================
def swap_video(input_video, source_img, output_video):
    source_latent = get_latent(source_img)

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(3))
    h = int(cap.get(4))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob, M, face = prepare_target(frame)
        if blob is None:
            writer.write(frame)
            continue

        # ONNX inference
        outputs = session.run(
            [output_name],
            {input_target_name: blob, input_source_name: source_latent}
        )
        swapped = Image.postprocess_face(outputs[0])

        # blend back
        blended = Image.blend_swapped_image(swapped, frame, M)

        writer.write(blended)

    cap.release()
    writer.release()

# ====================
# RUN
# ====================
swap_video(
    input_video="input.mp4",
    source_img="source.jpg",
    output_video="output_reswapper.mp4"
)
