import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from groundingdino.util.inference import load_image, predict
from torchvision.ops import box_iou

# ============================================================
# BATCH SETTINGS (18 Feb)
# ============================================================
IMG_DIR   = "/content"                 # <-- folder of input images
OUT_DIR   = "/content/hflip_outputs_larger_smaller"   # <-- folder to save outputs
EXTS      = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

BOX_THRESHOLD  = 0.15
TEXT_THRESHOLD = 0.15
IOU_THR_DUP    = 0.50

FONT_SCALE     = 0.5
BOX_THICKNESS  = 2
TEXT_THICKNESS = 2



os.makedirs(OUT_DIR, exist_ok=True)



# ============================================================
# HELPERS
# ============================================================
def sigmoid_logits(logits):
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def boxes_to_xyxy_pixels(boxes, W, H):
    b = to_numpy(boxes).astype(np.float32)

    if len(b) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # GroundingDINO often outputs normalized cxcywh
    if np.max(b) <= 1.5:
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        b = np.stack([x1, y1, x2, y2], axis=1)

    b[:, 0] = np.clip(b[:, 0], 0, W - 1)
    b[:, 1] = np.clip(b[:, 1], 0, H - 1)
    b[:, 2] = np.clip(b[:, 2], 0, W - 1)
    b[:, 3] = np.clip(b[:, 3], 0, H - 1)
    return b

# ============================================================
# PRIORITY-AWARE IoU DEDUP
# ============================================================
def iou_dedup_priority(boxes, scores, phrases, iou_thr=0.5):
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    cps = [canonical_phrase(p) or "default" for p in phrases]
    prs = np.array([PHRASE_PRIORITY.get(cp, 0) for cp in cps], dtype=np.int32)

    # sort by priority desc, then score desc
    order = np.lexsort((-scores, -prs))

    keep = []
    for idx in order:
        if not keep:
            keep.append(idx)
            continue

        ious = box_iou(
            torch.tensor(boxes[idx]).unsqueeze(0),
            torch.tensor(boxes[keep])
        ).squeeze(0).cpu().numpy()

        if float(np.max(ious)) < float(iou_thr):
            keep.append(idx)

    keep = np.array(keep, dtype=np.int64)
    return boxes[keep], scores[keep], [phrases[i] for i in keep]

# ============================================================
# DRAW
# ============================================================
def draw_boxes(bgr, boxes, scores, phrases):
    out = bgr.copy()
    placed = []

    def wrap_text(text, max_chars=30, max_lines=2):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if len(test) <= max_chars:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
                if len(lines) >= max_lines - 1:
                    break
        if cur and len(lines) < max_lines:
            lines.append(cur)
        if len(lines) > 0 and len(" ".join(lines).split()) < len(words):
            lines[-1] += " ..."
        return lines if lines else [text[:max_chars]]

    def overlap(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    for i in range(len(boxes)):
        cp = canonical_phrase(phrases[i])
        if cp is None:
            continue

        x1, y1, x2, y2 = map(int, boxes[i])
        color = PHRASE_COLORS.get(cp, PHRASE_COLORS["default"])

        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        original_phrase = str(phrases[i])  # keep EXACT phrase
        lines = wrap_text(original_phrase, 30, 2)
        lines[-1] = f"{lines[-1]} {float(scores[i]):.2f}"

        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS) for t in lines]
        line_height = max(h + b for ((w, h), b) in sizes) + 3
        block_height = line_height * len(lines)
        max_width = max(w for ((w, h), b) in sizes) if sizes else 10

        lx = x1
        ly_top = y1 - block_height - 6
        if ly_top < 0:
            ly_top = y1 + 6

        r = (lx, ly_top, lx + max_width, ly_top + block_height)
        while any(overlap(r, prev) for prev in placed):
            ly_top += block_height + 4
            r = (lx, ly_top, lx + max_width, ly_top + block_height)

        placed.append(r)

        cv2.rectangle(out, (r[0], r[1]), (r[2], r[3]), (0, 0, 0), -1)

        y_cursor = ly_top + line_height - 3
        for t in lines:
            cv2.putText(out, t, (lx, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE, color,
                        TEXT_THICKNESS, cv2.LINE_AA)
            y_cursor += line_height

    return out

# ============================================================
# PROCESS ONE IMAGE
# ============================================================
def process_one_image(image_path: str, out_path: str):
    image_source, image = load_image(image_path)  # image_source: RGB HxWx3, image: torch tensor for model
    H, W = image_source.shape[:2]

    # flip for inference + visualization
    image_flip = TF.hflip(image)
    image_source_flip = np.ascontiguousarray(image_source[:, ::-1, :])
    bgr_flip = cv2.cvtColor(image_source_flip, cv2.COLOR_RGB2BGR)

    # predict
    boxes, logits, phrases = predict(
        model=model,
        image=image_flip,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    scores = sigmoid_logits(logits)
    boxes = boxes_to_xyxy_pixels(boxes, W, H)

    # keep only "hole ladder"
    keep_boxes, keep_scores, keep_phrases = [], [], []
    for b, s, p in zip(boxes, scores, phrases):
        if canonical_phrase(p) is None:
            continue
        keep_boxes.append(b)
        keep_scores.append(float(s))
        keep_phrases.append(p)

    # dedup
    if len(keep_boxes) > 0:
        boxes_dd, scores_dd, phrases_dd = iou_dedup_priority(
            keep_boxes, keep_scores, keep_phrases, iou_thr=IOU_THR_DUP
        )
    else:
        boxes_dd = np.zeros((0, 4), dtype=np.float32)
        scores_dd = np.zeros((0,), dtype=np.float32)
        phrases_dd = []

    # draw + save
    bgr_out = draw_boxes(bgr_flip, boxes_dd, scores_dd, phrases_dd)
    ok = cv2.imwrite(out_path, bgr_out)
    return ok, len(keep_boxes), len(boxes_dd)

# ============================================================
# RUN BATCH
# ============================================================
files = [
    f for f in os.listdir(IMG_DIR)
    if os.path.isfile(os.path.join(IMG_DIR, f)) and f.lower().endswith(EXTS)
]
files.sort()

print(f"Found {len(files)} images in: {IMG_DIR}")
print("Saving to:", OUT_DIR)

done = 0
for fname in files:
    in_path = os.path.join(IMG_DIR, fname)
    base, ext = os.path.splitext(fname)
    out_path = os.path.join(OUT_DIR, f"hflip_{base}.jpg")  # always jpg output

    ok, n_before, n_after = process_one_image(in_path, out_path)
    if ok:
        done += 1
        print(f"[OK] {fname}  | kept={n_before}  dedup={n_after}  -> {out_path}")
    else:
        print(f"[FAIL] {fname} -> {out_path}")

print(f"Done. Saved {done}/{len(files)} images.")
