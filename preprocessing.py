import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# Device defined HERE
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225]
    )
])

class_names = ["Normal", "Shoplifting"]


@torch.no_grad()
def multi_clip_predict(
    model,
    video_path,
    num_clips=5,
    num_frames_per_clip=32,
    alpha=4
):
    model.eval()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_starts = np.linspace(
        0,
        max(0, total_frames - num_frames_per_clip),
        num_clips
    ).astype(int)

    probs_all = []

    for start in clip_starts:
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for _ in range(num_frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame)
            frames.append(frame)

        if len(frames) == 0:
            continue

        while len(frames) < num_frames_per_clip:
            frames.append(frames[-1])

        frames = torch.stack(frames)
        fast = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        slow = fast[:, :, ::alpha, :, :]

        logits = model([slow, fast])
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs_all.append(probs)

    cap.release()

    if len(probs_all) == 0:
        return None, None

    avg_probs = np.mean(probs_all, axis=0)
    pred_class = int(np.argmax(avg_probs))

    return pred_class, avg_probs
