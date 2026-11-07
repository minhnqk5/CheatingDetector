import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from config import *
from motion import Motion, person_visual
def associate_faces(boxes, faces):
    """Conservative one-to-one containment. Never give one face to two people."""
    assigned = {}
    for face in faces:
        x, y, w, h = face[:4]
        cx, cy = x+w/2, y+h/2
        owners = [i for i, (a,b,c,d) in enumerate(boxes) if a <= cx <= c and b <= cy <= b + .6*(d-b)]
        if len(owners) == 1:
            assigned.setdefault(owners[0], []).append(face)
    return {i:fs[0] for i,fs in assigned.items() if len(fs)==1}


def box_iou(a, b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def match_pose_people(person_boxes, pose_boxes):
    """One-to-one, mutual best match; reject close runner-up matches."""
    scores=[[box_iou(a,b) for b in pose_boxes] for a in person_boxes]
    out={}
    for i,row in enumerate(scores):
        if not row:continue
        ranked=sorted(range(len(row)),key=lambda j:row[j],reverse=True)
        j=ranked[0]
        if row[j]<.45 or (len(ranked)>1 and row[j]-row[ranked[1]]<.15):continue
        column=sorted(range(len(scores)),key=lambda k:scores[k][j],reverse=True)
        if column[0]!=i or (len(column)>1 and scores[i][j]-scores[column[1]][j]<.15):continue
        out[i]=j
    return out


def head_observation(points, min_conf=.45, threshold=.22, sign=1.):
    """Heuristic 2D cues with explicit rejection reasons, not calibrated yaw."""
    result=dict(raw_direction='POSE_UNKNOWN', turn_ratio=None,
                head_method='none', head_reason='NO_POSE_MATCH', head_confidence=0.0)
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[0]<7 or points.shape[1]<3:
        return result
    def valid(k):
        return bool(np.isfinite(points[k,:3]).all() and points[k,2]>=min_conf
                    and (points[k,:2] != 0).any())
    def reject(reason):
        result['head_reason']=reason
        return result
    def accept(ratio, method, indices):
        ratio=float(ratio)*sign
        direction='LEFT' if ratio>=threshold else 'RIGHT' if ratio<=-threshold else 'CENTER'
        return dict(raw_direction=direction,turn_ratio=ratio,head_method=method,
                    head_reason='OK',head_confidence=float(min(points[k,2] for k in indices)))
    if not valid(0):return reject('LOW_NOSE_CONF')
    # UP is an image-space cue, not a calibrated pitch angle. Require eyes,
    # ears and nose; missing landmarks must not become an up alert.
    if all(valid(k) for k in (0,1,2,3,4)):
        ears=points[[3,4],:2];eyes=points[[1,2],:2]
        ear_span=float(np.linalg.norm(ears[0]-ears[1]))
        if ear_span>=12 and abs(ears[0,1]-ears[1,1])<ear_span*.25:
            ear_mid=ears.mean(axis=0);eye_mid=eyes.mean(axis=0);nose=points[0,:2]
            if nose[1]<ear_mid[1]-.10*ear_span and nose[1]<=eye_mid[1]+.05*ear_span and abs(nose[0]-ear_mid[0])<ear_span*.3:
                return dict(raw_direction='UP',turn_ratio=None,head_method='raised_nose_eyes_ears',
                            head_reason='UP_CUE',head_confidence=float(min(points[k,2] for k in (0,1,2,3,4))))
    # A profile can have only one reliable ear. Do not require the nose to be
    # off the SHOULDER midpoint: head and torso can rotate together.
    profile = None
    if valid(5) and valid(6):
        sa,sb=points[[5,6],:2]
        if sa[0]>sb[0]:sa,sb=sb,sa
        axis=sb-sa; width=float(np.linalg.norm(axis))
        if width>=24 and abs(axis[1])<width*.5:
            unit=axis/width; down=np.array([-unit[1],unit[0]])
            nose=points[0,:2];middle=(sa+sb)/2
            visible=[k for k in (3,4) if valid(k)]
            if len(visible)==1 and float(np.dot(nose-middle,down))<-.08*width:
                ear_id=visible[0];eye_id=1 if ear_id==3 else 2
                if valid(eye_id):
                    ear=points[ear_id,:2];eye=points[eye_id,:2]
                    ne=float(np.dot(nose-ear,unit))/width
                    nx=float(np.dot(nose-eye,unit))/width
                    ex=float(np.dot(eye-ear,unit))/width
                    vertical=float(np.dot(nose-eye,down))/width
                    if (ne*nx>0 and ne*ex>0 and .24<=abs(ne)<=.8 and
                        .035<=abs(nx)<=.35 and -.1<=vertical<=.18):
                        profile=accept(ne,'single_side_profile',(0,eye_id,ear_id,5,6))

    # Evaluate BOTH pairs. Ear and eye spans have different scales; normalize
    # each cue by its own turn threshold before combining them.
    cues=[]; geometry_rejected=False
    for pair,limit,max_down in (((3,4),threshold,.75),((1,2),max(.30,threshold*1.5),1.4)):
        if not all(valid(k) for k in pair):continue
        a,b=points[list(pair),:2]
        if a[0]>b[0]:a,b=b,a
        axis=b-a;span=float(np.linalg.norm(axis))
        if span<8 or abs(axis[1])>span*.5:continue
        unit=axis/span;down=np.array([-unit[1],unit[0]])
        delta=points[0,:2]-(a+b)/2
        vertical=float(np.dot(delta,down))/span
        # Large downward displacement is ambiguous, not evidence of yaw.
        if vertical>max_down or vertical<-.5:
            geometry_rejected=True;continue
        ratio=float(np.dot(delta,unit))/span
        if abs(ratio)>1.2:continue
        confidence=float(min(points[k,2] for k in (0,*pair)))
        cues.append((ratio/max(limit,1e-6),confidence,pair,ratio))
    if profile is not None:
        # A strong contradictory bilateral cue must not be silently ignored.
        signed_profile=profile['turn_ratio']
        if any(abs(score)>=1 and score*sign*signed_profile<0 for score,_,_,_ in cues):
            return reject('CONFLICTING_PROFILE_CUES')
        return profile
    if not cues:
        return reject('DOWN_OR_TILTED' if geometry_rejected else 'LOW_HEAD_SUPPORT')
    if len(cues)==2:
        first,second=cues
        if first[0]*second[0]<0 and min(abs(first[0]),abs(second[0]))>.55:
            return reject('CONFLICTING_HEAD_CUES')
        score=sum(c[0]*c[1] for c in cues)/sum(c[1] for c in cues)
        # Strong eye evidence should not disappear just because the ear pair
        # is near center, provided both cues point to the same side.
        stronger=max(cues,key=lambda c:abs(c[0]))
        weaker=min(cues,key=lambda c:abs(c[0]))
        if abs(stronger[0])>=1.6 and abs(weaker[0])>=.35 and stronger[0]*weaker[0]>0:
            score=float(np.sign(stronger[0]))*max(1.01,abs(score))
        indices=(0,1,2,3,4);method='eyes_ears_consensus'
    else:
        score,_,pair,_=cues[0]
        indices=(0,*pair);method='ears_pair' if pair==(3,4) else 'eyes_pair'
    observation=accept(score*threshold,method,indices)
    # An incomplete face near center does not establish a straight gaze.
    if observation['raw_direction']=='CENTER' and len(cues)==1 and geometry_rejected:
        return reject('PARTIAL_DOWNWARD_FACE')
    observation['head_cue_scores']=[round(float(c[0]*sign),3) for c in cues]
    return observation


def select_head_observation(full, cropped):
    """Fuse crop/full directions; never hide disagreement as confident CENTER."""
    a=full['raw_direction'];b=cropped['raw_direction']
    if b=='POSE_UNKNOWN':return full,False
    if a=='POSE_UNKNOWN':return cropped,True
    # Preserve existing UP handling, outside the left/right change.
    if a=='UP' or b=='UP':return cropped,True
    opposite=a in ('LEFT','RIGHT') and b in ('LEFT','RIGHT') and a!=b
    if opposite:
        unknown=dict(full,raw_direction='POSE_UNKNOWN',head_reason='CROP_DIRECTION_CONFLICT')
        return unknown,False
    if a==b:return (cropped,True) if cropped['head_confidence']>=full['head_confidence'] else (full,False)
    if a=='CENTER' and b in ('LEFT','RIGHT'):
        if cropped['head_confidence']>=full['head_confidence']-.05:return cropped,True
        return dict(full,raw_direction='POSE_UNKNOWN',head_reason='CROP_LOW_CONF_CONFLICT'),False
    return full,False


def keypoint_head_direction(points,min_conf=.45,threshold=.22,sign=1.):
    info=head_observation(points,min_conf,threshold,sign)
    return info['raw_direction'],info['turn_ratio']


def summarize_points(points):
    array=np.asarray(points,dtype=float)
    if array.ndim!=2 or array.shape[1]<3:return []
    return [[round(float(v),3) if np.isfinite(v) else 0.0 for v in row[:3]]
            for row in array[:7]]

class Faces:

    def __init__(self):
        print('Loading YuNet...')
        self.det = cv2.FaceDetectorYN.create(str(FACE_DETECTOR_MODEL), '', (320, 320), 0.6, 0.3, 5000)
        self.det_lock = threading.Lock()
    def faces(self, image):
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        max_size = FACE_MAX_SIZE
        scale = min(1.0, max_size / max(width, height))
        if scale < 1.0:
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = image
        h, w = small.shape[:2]
        try:
            with self.det_lock:
                self.det.setInputSize((w, h))
                _, detected = self.det.detect(small)
            if detected is None:
                return []
            detected = detected.copy()
            detected[:, :4] /= scale
            detected[:, 4:14] /= scale
            return detected
        except Exception as e:
            print(f'[FACE DETECT ERROR] {type(e).__name__}: {e}')
            raise

class Pose:

    def __init__(self):
        self.temporal = {}
        self.smooth = {}
        self.direction_state = {}
        self.pending_direction = {}
        self.pending_count = {}

    def estimate(self, face, width, height, track_id=None):
        if face is None or len(face) < 14 or width <= 0 or (height <= 0):
            return None
        right_eye = np.array([face[4], face[5]], dtype=np.float64)
        left_eye = np.array([face[6], face[7]], dtype=np.float64)
        nose = np.array([face[8], face[9]], dtype=np.float64)
        right_mouth = np.array([face[10], face[11]], dtype=np.float64)
        left_mouth = np.array([face[12], face[13]], dtype=np.float64)
        points_2d = np.array([right_eye, left_eye, nose, right_mouth, left_mouth], dtype=np.float64)
        if not np.isfinite(points_2d).all():
            return None
        eye_distance = np.linalg.norm(right_eye - left_eye)
        if eye_distance < 14:
            return None
        # YuNet order is anatomical right eye, left eye, nose, right mouth, left mouth.
        # Anatomical right landmarks appear on image-left for a frontal face, so their
        # model X coordinate must be negative in OpenCV's camera coordinate system.
        model_points = np.array([
            [-32.0, -30.0, -26.0],  # right eye
            [ 32.0, -30.0, -26.0],  # left eye
            [  0.0,   0.0,   0.0],  # nose tip
            [-25.0,  30.0, -18.0],  # right mouth corner
            [ 25.0,  30.0, -18.0],  # left mouth corner
        ], dtype=np.float64)
        focal_length = float(max(width, height))
        camera_matrix = np.array([[focal_length, 0, width / 2.0], [0, focal_length, height / 2.0], [0, 0, 1]], dtype=np.float64)
        distortion = np.zeros((4, 1), dtype=np.float64)
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, points_2d, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
        except Exception as e:
            print(f'[POSE ERROR] {e}')
            return None
        if not success or translation_vector[2, 0] <= 0:
            return None
        projected, _ = cv2.projectPoints(model_points, rotation_vector, translation_vector,
                                        camera_matrix, distortion)
        error = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-points_2d)**2, axis=1)))
        if not np.isfinite(error) or error > eye_distance * 0.18:
            return None
        try:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        except Exception as e:
            print(f'[RODRIGUES ERROR] {e}')
            return None
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-06
        if not singular:
            yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
            pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
            roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        else:
            yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
            pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
            roll = 0.0
        yaw = float(yaw) * float(HEAD_YAW_SIGN)
        pitch = float(pitch)
        roll = float(roll)
        # Reject implausible solutions from tiny, occluded, or badly matched faces.
        # These frames become POSE_UNKNOWN and cannot trigger a left/right alert.
        if abs(yaw) > 75.0 or abs(pitch) > 60.0 or abs(roll) > 45.0:
            return None
        if track_id is not None:
            previous = self.smooth.get(track_id)
            alpha = float(HEAD_YAW_SMOOTH_ALPHA)
            if previous is None:
                smooth_yaw = yaw
                smooth_pitch = pitch
                smooth_roll = roll
            else:
                smooth_yaw = alpha * yaw + (1.0 - alpha) * previous['yaw']
                smooth_pitch = (previous['pitch'] + alpha * ((pitch-previous['pitch']+180)%360-180)+180)%360-180
                smooth_roll = alpha * roll + (1.0 - alpha) * previous['roll']
            self.smooth[track_id] = {'yaw': smooth_yaw, 'pitch': smooth_pitch, 'roll': smooth_roll}
            yaw = float(smooth_yaw)
            pitch = float(smooth_pitch)
            roll = float(smooth_roll)
        return {'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll)}

    def forget(self,track_id):
        for state in (self.temporal,self.smooth,self.direction_state,self.pending_direction,self.pending_count):
            state.pop(track_id,None)

    def stable_raw(self,candidate,track_id):
        if track_id < 0:return 'POSE_UNKNOWN'
        if candidate in ('POSE_UNKNOWN','CENTER'):
            self.direction_state[track_id]=candidate
            self.pending_count[track_id]=0
            self.pending_direction[track_id]=candidate
            return candidate
        previous=self.direction_state.get(track_id,'CENTER')
        if candidate==previous:
            self.pending_direction[track_id]=candidate
            self.pending_count[track_id]=0
            return candidate
        if self.pending_direction.get(track_id)!=candidate:
            self.pending_count[track_id]=1
            self.pending_direction[track_id]=candidate
        else:self.pending_count[track_id]=self.pending_count.get(track_id,0)+1
        if self.pending_count[track_id]>=HEAD_DIRECTION_STABLE_FRAMES:
            self.direction_state[track_id]=candidate
            self.pending_count[track_id]=0
            return candidate
        return 'PENDING'

    def direction(self, yaw, pitch, track_id=None):
        if yaw is None or pitch is None:
            return 'POSE_UNKNOWN'
        yaw = float(yaw)
        pitch = float(pitch)
        # Pitching down/up distorts the 5-point yaw estimate. Do not interpret
        # these frames as left/right; clear any pending direction immediately.
        if not np.isfinite(yaw) or not np.isfinite(pitch) or abs(pitch) > HEAD_YAW_MAX_ABS_PITCH:
            if track_id is not None:
                self.direction_state[track_id] = 'CENTER'
                self.pending_direction[track_id] = 'CENTER'
                self.pending_count[track_id] = 0
            return 'POSE_UNKNOWN'
        threshold = float(HEAD_YAW_THRESHOLD)
        center_threshold = float(HEAD_YAW_CENTER_THRESHOLD)
        previous = self.direction_state.get(track_id, 'CENTER') if track_id is not None else 'CENTER'
        # Positive yaw means the candidate turns to their own left under the
        # corrected YuNet landmark/model-point correspondence.
        if yaw <= -threshold:
            raw = 'RIGHT'
        elif yaw >= threshold:
            raw = 'LEFT'
        else:
            raw = 'CENTER'
        if previous == 'RIGHT' and yaw < -center_threshold:
            candidate = 'RIGHT'
        elif previous == 'LEFT' and yaw > center_threshold:
            candidate = 'LEFT'
        else:
            candidate = raw
        if track_id is None:
            return candidate
        if candidate == previous:
            self.pending_direction[track_id] = candidate
            self.pending_count[track_id] = 0
            return previous
        if candidate == 'CENTER':
            self.direction_state[track_id] = 'CENTER'
            self.pending_direction[track_id] = 'CENTER'
            self.pending_count[track_id] = 0
            return 'CENTER'
        pending = self.pending_direction.get(track_id)
        if pending != candidate:
            self.pending_direction[track_id] = candidate
            self.pending_count[track_id] = 1
        else:
            self.pending_count[track_id] = self.pending_count.get(track_id, 0) + 1
        if self.pending_count[track_id] >= HEAD_DIRECTION_STABLE_FRAMES:
            self.direction_state[track_id] = candidate
            self.pending_count[track_id] = 0
            return candidate
        return previous

    def temporal_update(self, track_id, direction, media_time=None):
        """
        Chỉ LEFT/RIGHT được tính; không đủ dữ liệu sẽ reset bộ đếm.
        - CENTER/POSE_UNKNOWN: kết thúc phiên, đưa bộ đếm về 0.
        - LEFT/RIGHT: tiếp tục bộ đếm hiện tại.
        - Webcam: dùng time.perf_counter().
        - Video: dùng media_time.
        """
        if track_id is None or track_id < 0:
            return (0.0, False, False)
        now = float(media_time) if media_time is not None else time.perf_counter()
        suspicious = direction in ('LEFT', 'RIGHT')
        state = self.temporal.get(track_id)
        if not suspicious:
            self.temporal[track_id] = {'start': None, 'last_time': now, 'event_sent': False}
            return (0.0, False, False)
        if state is None or state.get('start') is None:
            state = {'start': now, 'last_time': now, 'event_sent': False}
            self.temporal[track_id] = state
        elif now < state['last_time']:
            state['start'] = now
            state['event_sent'] = False
        state['last_time'] = now
        duration = max(0.0, now - state['start'])
        long_event = duration >= LOOKING_AWAY_SECONDS
        should_log_event = long_event and (not state['event_sent'])
        if should_log_event:
            state['event_sent'] = True
        return (float(duration), bool(long_event), bool(should_log_event))

    def remove_lost(self, active_ids):
        active_ids = set(active_ids)
        stores = [self.temporal, self.smooth, self.direction_state, self.pending_direction, self.pending_count]
        for store in stores:
            for track_id in list(store.keys()):
                if track_id not in active_ids:
                    del store[track_id]

class Monitor:

    def __init__(self, db):
        # No student database or identity recognition in this version.
        print('Loading YOLO model...')
        self.model = YOLO(str(MODEL_PATH))
        # TensorRT engines are already optimized/fused at export time.
        if MODEL_PATH.suffix.lower() == '.pt':
            self.model.fuse(verbose=False)
        print('YOLO model loaded.')
        if HEAD_BACKEND not in ('pose','yunet'):
            raise ValueError('HEAD_BACKEND phải là pose hoặc yunet')
        self.track_history = {}
        self.motion = Motion()
        self.faces = Faces() if HEAD_BACKEND == 'yunet' else None
        self.head_model = YOLO(str(POSE_MODEL_PATH)) if HEAD_BACKEND == 'pose' else None
        self.pose = Pose()
        self.cache = {}
        self.lastrec = {}
        self.events = []
        self.objects = []
        self.last_tracks = []
        self.persons = 0
        self.recognized = 0
        self.fps = 0.0
        self.ai_fps = 0.0
        self.prev = time.perf_counter()
        self._event_times = {}
        self.frame_count = 0

    def retry_person_pose(self,frame,person_boxes,index):
        """Crop retries improve sampling resolution; ambiguous owners stay unknown."""
        h,w=frame.shape[:2]; target=person_boxes[index]
        a,b,c,d=map(float,target); pad=.06*max(c-a,d-b)
        x1=max(0,int(a-pad));y1=max(0,int(b-pad))
        x2=min(w,int(c+pad));y2=min(h,int(d+pad))
        if x2-x1<24 or y2-y1<24:return None
        results=self.head_model.predict(frame[y1:y2,x1:x2],imgsz=POSE_CROP_SIZE,
            conf=.25,device=DEVICE,half=HALF,verbose=False)
        if not results or results[0].keypoints is None or results[0].boxes is None:return None
        boxes=results[0].boxes.xyxy.cpu().numpy().copy()
        points=results[0].keypoints.data.cpu().numpy().copy()
        if len(boxes)!=len(points):return None
        boxes[:,[0,2]]+=x1;boxes[:,[1,3]]+=y1
        points[:,:,0]+=x1;points[:,:,1]+=y1
        assignment=match_pose_people(person_boxes,boxes)
        j=assignment.get(index)
        if j is None:return None
        row=points[j]
        if row.shape[1]<3 or row[0,2]<POSE_KEYPOINT_CONFIDENCE:return None
        nx,ny=row[0,:2]
        owners=[i for i,(l,t,r,bottom) in enumerate(person_boxes)
                if l<=nx<=r and t<=ny<=t+.6*(bottom-t)]
        if owners!=[index]:return None
        return row

    def process(self, frame, media_time=None):
        if frame is None:
            return self.last_tracks
        now = time.perf_counter()
        dt = max(now - self.prev, 1e-06)
        camera_fps = 1.0 / dt
        if self.fps == 0:
            self.fps = camera_fps
        else:
            self.fps = self.fps * 0.9 + camera_fps * 0.1
        self.prev = now
        self.frame_count += 1
        if (self.frame_count - 1) % PROCESS_EVERY_N_FRAMES != 0:
            return self.last_tracks
        ai_start = time.perf_counter()
        try:
            results = self.model.track(frame, persist=True, tracker=TRACKER_PATH, classes=DETECTION_CLASSES, conf=CONFIDENCE, iou=IOU, imgsz=IMG_SIZE, device=DEVICE, half=HALF, verbose=False)
        except Exception as e:
            message = str(e)
            if 'out of memory' in message.lower():
                raise RuntimeError('GPU hết VRAM. Dừng ứng dụng, giảm AI_IMAGE_SIZE hoặc chọn model nhỏ hơn bằng YOLO_MODEL; xem README.') from e
            raise RuntimeError('YOLO inference failed: '+message[:300]) from e
        if not results:
            raise RuntimeError("YOLO returned no result")
        result = results[0]
        if result.boxes is None:
            self.pose.remove_lost([])
            self.track_history.clear()
            self.motion.retain([])
            self.last_tracks = []
            self.objects = []
            self.persons = 0
            self.recognized = 0
            return self.last_tracks
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(xyxy)
        tracks = []
        objects = []
        active_ids = []
        yolo_done = time.perf_counter()
        self.persons = 0
        self.recognized = 0
        height, width = frame.shape[:2]
        # One YuNet pass per AI sample; ambiguous overlaps get no head pose.
        detected_faces = self.faces.faces(frame) if self.faces and any(classes == 0) else []
        pose_rows = {}
        if self.head_model is not None and any(classes == 0):
            head_results = self.head_model.predict(frame, imgsz=POSE_IMAGE_SIZE,
                conf=.35, device=DEVICE, half=HALF, verbose=False)
            if head_results and head_results[0].keypoints is not None and head_results[0].boxes is not None:
                head_result = head_results[0]
                pose_boxes = head_result.boxes.xyxy.cpu().numpy()
                points = head_result.keypoints.data.cpu().numpy()
                pboxes = [box for box, cls in zip(xyxy, classes) if cls == 0]
                pose_rows = {i:points[j] for i,j in match_pose_people(pboxes,pose_boxes).items()}
        person_boxes = [box for box, cls in zip(xyxy, classes) if cls == 0]
        face_assignments = associate_faces(person_boxes, detected_faces)
        prior_ids=set(self.track_history)
        person_index = -1
        for box, confidence, class_id, track_id in zip(xyxy, confs, classes, track_ids):
            if class_id != 0:
                continue
            person_index += 1
            self.persons += 1
            track_id = int(track_id)
            if track_id >= 0:
                active_ids.append(track_id)
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            crop = frame[y1:y2, x1:x2]
            item = {'track_id': int(track_id), 'bbox': bbox, 'student_id': None, 'name': 'Chưa nhận diện', 'seat': None, 'similarity': 0.0, 'yaw': None, 'pitch': None, 'roll': None, 'direction': 'CENTER', 'duration': 0.0, 'long_event': False}
            timestamp = float(media_time) if media_time is not None else time.monotonic()
            face = face_assignments.get(person_index)
            direction = 'POSE_UNKNOWN'
            row=[]
            previous=self.track_history.get(track_id)
            motion_reset=previous is None
            if previous is not None:
                old_time,old_box=previous
                from rules import continuous_box
                if timestamp<old_time or timestamp-old_time>HEAD_STATE_MAX_GAP or not continuous_box(old_box,bbox):
                    self.pose.forget(track_id)
                    motion_reset=True
            if track_id>=0:self.track_history[track_id]=(timestamp,bbox)
            if self.head_model is not None:
                row=pose_rows.get(person_index,[])
                info=head_observation(row,POSE_KEYPOINT_CONFIDENCE,HEAD_TURN_RATIO,HEAD_YAW_SIGN)
                source='full_frame'; retry_error=None
                if (info['raw_direction'] in ('POSE_UNKNOWN','CENTER') or info.get('head_method') in ('single_side_profile','eyes_pair')) and POSE_CROP_RETRY:
                    try:
                        retry=self.retry_person_pose(frame,person_boxes,person_index)
                        if retry is not None:
                            retry_info=head_observation(retry,POSE_KEYPOINT_CONFIDENCE,HEAD_TURN_RATIO,HEAD_YAW_SIGN)
                            info,use_crop=select_head_observation(info,retry_info)
                            if use_crop:row=retry;source='person_crop'
                    except Exception as exc:
                        # Keep full-frame result; diagnostics expose crop failure.
                        retry_error=str(exc)[:160]
                raw=info['raw_direction']
                direction=self.pose.stable_raw(raw,track_id)
                item.update(info, head_source='body_pose', pose_source=source,
                            head_keypoints=summarize_points(row),
                            stable_samples=(HEAD_DIRECTION_STABLE_FRAMES if direction in ('LEFT','RIGHT') else self.pose.pending_count.get(track_id,0)),
                            pose_retry_error=retry_error)
                if direction=='PENDING':item['head_reason']='WAIT_STABLE'
            elif face is not None:
                pose = self.pose.estimate(face, width, height, track_id)
                if pose is not None:
                    item.update(pose)
                    direction = self.pose.direction(pose['yaw'], pose['pitch'], track_id)
            item.update(self.motion.update(track_id,pose_rows.get(person_index,[]),bbox,timestamp,reset=motion_reset))
            duration, long_event, should_log_event = self.pose.temporal_update(track_id, direction, media_time)
            item.update({'direction': direction, 'duration': float(duration), 'long_event': bool(long_event)})
            if should_log_event:
                if direction == 'POSE_UNKNOWN':
                    event_type = 'POSE_UNKNOWN_LONG'
                else:
                    event_type = f'LOOKING_{direction}_LONG'
                self._event(event_type, item, media_time=media_time, duration=round(duration, 2), direction=direction)
            tracks.append(item)
        face_done = time.perf_counter()
        self.pose.remove_lost(active_ids)
        self.motion.retain(active_ids)
        for lost_id in prior_ids-set(active_ids):self.track_history.pop(lost_id,None)
        for cache in (self.cache, self.lastrec):
            for lost_id in set(cache) - set(active_ids):
                cache.pop(lost_id, None)
        for box, confidence, class_id, object_id in zip(xyxy, confs, classes, track_ids):
            class_id = int(class_id)
            if class_id not in (63, 67, 73) or confidence < OBJECT_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            nearby_person = None
            class_name = result.names.get(class_id, str(class_id))
            obj = {'track_id': int(object_id), 'class_id': int(class_id), 'class_name': class_name, 'confidence': float(confidence), 'bbox': bbox, 'student_id': nearby_person['student_id'] if nearby_person else None, 'name': nearby_person['name'] if nearby_person else 'Chưa nhận diện', 'seat': nearby_person['seat'] if nearby_person else None}
            objects.append(obj)
            if class_id == 67:
                self._event('PHONE_DETECTED', obj, media_time=media_time)
            elif class_id == 73:
                self._event('BOOK_DETECTED', obj, media_time=media_time)
            elif class_id == 63:
                self._event('LAPTOP_DETECTED', obj, media_time=media_time)
        self.objects = objects
        self.last_tracks = tracks
        self.timings = {'yolo_ms': round((yolo_done-ai_start)*1000,1), 'face_ms': round((face_done-yolo_done)*1000,1)}
        ai_time = time.perf_counter() - ai_start
        if ai_time > 0:
            current_ai_fps = 1.0 / ai_time
            if self.ai_fps == 0:
                self.ai_fps = current_ai_fps
            else:
                self.ai_fps = self.ai_fps * 0.8 + current_ai_fps * 0.2
        return self.last_tracks

    def _event(self, event_type, obj, media_time=None, **extra):
        now = time.time()
        key = (event_type, obj.get('track_id'))
        cooldown_time = float(media_time) if media_time is not None else now
        last = self._event_times.get(key)
        if last is not None and cooldown_time - last < 3.0:
            return
        self._event_times[key] = cooldown_time
        event = {'type': str(event_type), 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'video_time': round(float(media_time), 2) if media_time is not None else None, 'student_id': obj.get('student_id'), 'name': obj.get('name'), 'seat': obj.get('seat'), **extra}
        self.events.insert(0, event)
        self.events = self.events[:200]

    def near(self, person_bbox, object_bbox):
        person_cx = (person_bbox[0] + person_bbox[2]) / 2.0
        person_cy = (person_bbox[1] + person_bbox[3]) / 2.0
        object_cx = (object_bbox[0] + object_bbox[2]) / 2.0
        object_cy = (object_bbox[1] + object_bbox[3]) / 2.0
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        return abs(person_cx - object_cx) < person_width * 0.9 and abs(person_cy - object_cy) < person_height * 1.2

    def draw(self, frame, tracks):
        for person in tracks:
            x1, y1, x2, y2 = person['bbox']
            color, head_label, body_label = person_visual(person)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {person['track_id']} | {head_label} | {body_label}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for obj in self.objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{obj['class_name']} {obj['confidence']:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        cv2.putText(frame, f'Worker FPS: {self.fps:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f'AI FPS: {self.ai_fps:.1f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f'AI: 1/{PROCESS_EVERY_N_FRAMES} frames', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame
