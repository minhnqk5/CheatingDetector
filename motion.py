"""Conservative torso observations; no identity recognition or cheating verdict."""
import math
import numpy as np


def person_visual(person, mode='camera'):
    head = person.get('direction', 'POSE_UNKNOWN')
    reasons = []

    # Tùy chỉnh theo mode: Nếu chạy video thì bỏ qua POSE_UNKNOWN để không tô khung đỏ khi cúi viết bài
    if mode == 'video':
        valid_head_reasons = ('LEFT', 'RIGHT', 'UP', 'BACK', 'PENDING')
    else:
        valid_head_reasons = ('LEFT', 'RIGHT', 'UP', 'BACK', 'PENDING')

    if head in valid_head_reasons:
        reasons.append('HEAD_' + head)

    if person.get('left_initial_position'):
        reasons.append('LEFT_INITIAL_POSITION')

    if person.get('body_action'):
        reasons.append(person['body_action'])

    labels = {
        'LEFT': 'QUAY ĐẦU TRÁI',
        'RIGHT': 'QUAY ĐẦU PHẢI',
        'UP': 'NGỬA ĐẦU',
        'BACK': 'QUAY ĐẰNG SAU',
        'CENTER': 'NHÌN THẲNG',
        'PENDING': 'ĐANG XÁC NHẬN',
        'POSE_UNKNOWN': 'CHƯA XÁC ĐỊNH HƯỚNG'
    }

    body = (
        'RỜI KHỎI VỊ TRÍ'
        if person.get('left_initial_position')
        else {
            'BODY_TURN': 'XOAY NGƯỜI',
            'BODY_LEAN': 'NGHIÊNG / NGẢ NGƯỜI'
        }.get(
            person.get('body_action'),
            'TẠI VỊ TRÍ'
        )
    )

    return (
        (0, 0, 255) if reasons else (0, 220, 0),
        labels.get(head, 'CHƯA XÁC ĐỊNH HƯỚNG'),
        body
    )


class Motion:
    def __init__(self): 
        self.previous = {}

    def update(self, tid, points, bbox, now, reset=False):
        out = dict(left_initial_position=False, body_action=None, movement_detected=False,
                   motion_status='UNKNOWN', motion_pixels=0.0)
        state = self.previous.get(tid)
        if reset or (state and not 0 < now - state['last'] <= 1.5):
            state = None
            self.previous.pop(tid, None)
        row = np.asarray(points, dtype=float)

        def valid(i):
            return row.ndim == 2 and row.shape[0] > i and row.shape[1] >= 3 and np.isfinite(row[i, :3]).all() and row[i, 2] >= .65 and np.any(row[i, :2] != 0)

        if tid < 0: return out
        if not all(valid(i) for i in (5, 6)):
            if state: state['pending'] = {}; state['last'] = now
            return out

        shoulder = row[[5, 6], :2].mean(axis=0)
        span = float(np.linalg.norm(row[5, :2] - row[6, :2]))
        if span < 20: return out

        hip = row[[11, 12], :2].mean(axis=0) if all(valid(i) for i in (11, 12)) else None
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        if state is None:
            state = dict(last=now, samples=[], base=None, pending={})
            self.previous[tid] = state

        state['last'] = now

        if state['base'] is None:
            state['samples'].append((now, shoulder, span, hip, center))
            samples = state['samples'][-8:]; state['samples'] = samples
            if len(samples) < 3 or now - samples[0][0] < .4:
                out['motion_status'] = 'CALIBRATING'; return out
            shoulders = np.array([x[1] for x in samples]); base_s = np.median(shoulders, axis=0)
            base_span = float(np.median([x[2] for x in samples]))
            if np.max(np.linalg.norm(shoulders - base_s, axis=1)) > max(8, base_span * .12):
                state['samples'] = samples[-1:]; out['motion_status'] = 'CALIBRATING'; return out
            hips = [x[3] for x in samples if x[3] is not None]
            state['base'] = (base_s, base_span, np.median(hips, axis=0) if len(hips) >= 3 else None, np.median([x[4] for x in samples], axis=0))

        bs, bw, bh, bc = state['base']

        def confirmed(key, yes):
            if not yes: state['pending'].pop(key, None); return False
            entry = state['pending'].setdefault(key, [now, 0]); entry[1] += 1
            return now - entry[0] >= .5 and entry[1] >= 3

        ds = shoulder - bs; dc = center - bc
        departure = False; lean = False
        if hip is not None and bh is not None:
            dh = hip - bh
            tolerance = max(35., bw * .6)
            departure = (np.linalg.norm(ds) > tolerance and np.linalg.norm(dh) > tolerance and
                         np.linalg.norm(ds - dh) < bw * .35 and np.linalg.norm(dc) > tolerance * .6 and np.dot(ds, dc) > 0)
            a = bs - bh; b = shoulder - hip
            if np.linalg.norm(a) > 20 and np.linalg.norm(b) > 20:
                angle = math.degrees(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
                lean = angle >= 25 and (abs(ds[0] - dh[0]) > bw * .4 or ds[1] - dh[1] < -bw * .35)

        turn = span < bw * .55 and np.linalg.norm(ds) < bw * .65
        departed = confirmed('departure', departure)
        action = 'BODY_TURN' if confirmed('turn', turn) else None
        if confirmed('lean', lean): action = 'BODY_LEAN'

        out.update(left_initial_position=departed, body_action=action,
                   movement_detected=bool(departed or action), motion_status='OBSERVED',
                   motion_pixels=float(np.linalg.norm(ds)), initial_position=bs.tolist())
        return out

    def retain(self, ids): 
        self.previous = {k: v for k, v in self.previous.items() if k in ids}
