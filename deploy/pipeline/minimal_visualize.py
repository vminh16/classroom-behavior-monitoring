import cv2

from python.visualize import localize_label


BASE_COLOR = (72, 198, 92)
WARN_COLOR = (0, 215, 255)
ALERT_COLOR = (0, 0, 255)
BOX_THICKNESS = 1
BOX_INSET = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.38
FONT_THICKNESS = 1
TEXT_SHADOW_COLOR = (0, 0, 0)
TEXT_SHADOW_THICKNESS = 2


def _color_for_state(state_name):
    state_text = str(state_name or "NORMAL").upper()
    if "ALERT" in state_text:
        return ALERT_COLOR
    if "WARN" in state_text:
        return WARN_COLOR
    return BASE_COLOR


def _normalize_cls_results(cls_action_res):
    if cls_action_res is None:
        return {}
    if isinstance(cls_action_res, dict):
        return cls_action_res

    id_to_res = {}
    for item in cls_action_res:
        try:
            track_id, res = item
        except Exception:
            continue
        id_to_res[int(track_id)] = res
    return id_to_res


def _display_label(label_text):
    text = str(label_text or "").strip().lower()
    if not text:
        return ""
    if "normal" in text or "binh_thuong" in text:
        return ""
    return text.replace("_", " ")


def _resolve_label_text(track_id, res, labels, mot_score=None):
    tokens = ["ID {}".format(track_id)]
    score = None if mot_score is None else float(mot_score)
    if not isinstance(res, dict):
        if score is not None:
            tokens.append("{:.2f}".format(score))
        return " | ".join(tokens)

    cls_id = None
    if 'class' in res:
        cls_id = int(res.get('class', 0))
    if 'score' in res and res.get('score') is not None:
        score = float(res.get('score'))
    if cls_id is not None and cls_id < len(labels):
        raw_label = labels[cls_id]
        display_label = _display_label(localize_label(raw_label))
        if not display_label:
            display_label = _display_label(raw_label)
        if display_label:
            tokens.append(display_label)
    if score is not None:
        tokens.append("{:.2f}".format(score))
    return " | ".join(tokens)


def _inset_box(x1, y1, x2, y2, width, height):
    inset = BOX_INSET if (x2 - x1) > 24 and (y2 - y1) > 24 else 1
    x1 = max(0, min(width - 1, x1 + inset))
    y1 = max(0, min(height - 1, y1 + inset))
    x2 = max(0, min(width - 1, x2 - inset))
    y2 = max(0, min(height - 1, y2 - inset))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def _draw_tag(image, x, y, text, color):
    height, width = image.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE,
                                         FONT_THICKNESS)
    left = max(0, min(width - tw - 1, x))
    baseline_y = y - 6
    if baseline_y - th < 0:
        baseline_y = min(height - baseline - 1, y + th + 8)
    origin = (left, baseline_y)
    cv2.putText(image, text, origin, FONT, FONT_SCALE, TEXT_SHADOW_COLOR,
                TEXT_SHADOW_THICKNESS, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, FONT_SCALE, color, FONT_THICKNESS,
                cv2.LINE_AA)


def draw_minimal_video_result(image_rgb, result, labels):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mot_res = result.get('mot')
    if mot_res is None or len(mot_res.get('boxes', [])) == 0:
        return image

    height, width = image.shape[:2]
    id_to_res = _normalize_cls_results(result.get('cls_action'))
    labels = labels or ['normal', 'using_phone', 'sleeping']

    for mot_box in mot_res['boxes']:
        track_id = int(mot_box[0])
        mot_score = float(mot_box[2]) if len(mot_box) > 2 else None
        x1, y1, x2, y2 = [int(v) for v in mot_box[3:7]]
        x1, y1, x2, y2 = _inset_box(x1, y1, x2, y2, width, height)

        res = id_to_res.get(track_id)
        state_name = res.get('state', 'NORMAL') if isinstance(
            res, dict) else 'NORMAL'
        color = _color_for_state(state_name)
        text = _resolve_label_text(track_id, res, labels, mot_score=mot_score)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS,
                      cv2.LINE_AA)
        _draw_tag(image, x1, y1, text, color)

    return image
