"""Cảnh báo quan sát theo từng track; không tự kết luận gian lận."""
import math

DEFAULTS = dict(head_hold_seconds=0.0, hold_seconds=3.0, absence_seconds=10.0, cooldown_seconds=15.0,
                max_gap_seconds=2.0, pitch_enabled=False, pitch_center=0.0,
                pitch_threshold=25.0, book_enabled=False, laptop_enabled=False)


def continuous_box(previous, current):
    """Reject abrupt track jumps; this cannot detect every ByteTrack ID switch."""
    a,b,c,d=previous; x,y,z,w=current
    old_area=max(1,(c-a)*(d-b)); new_area=max(1,(z-x)*(w-y))
    distance=math.hypot((a+c-x-z)/2,(b+d-y-w)/2)
    scale=max(1,math.hypot(c-a,d-b),math.hypot(z-x,w-y))
    return distance/scale <= .45 and .3 <= new_area/old_area <= 3.3


class Rules:
    def __init__(self, settings=None):
        self.settings=dict(DEFAULTS, **(settings or {}))
        self.reset()

    def reset(self):
        self.pending={}; self.last={}; self.prev=None
        self.track_boxes={}; self.object_boxes={}
        self.alerted_tracks=set()

    def update(self, tracks, objects, seats, students, width, height, now):
        if self.prev is not None and (now < self.prev or now-self.prev>self.settings['max_gap_seconds']):
            self.pending.clear(); self.track_boxes.clear(); self.object_boxes.clear()
            if now < self.prev:self.last.clear()
        self.prev=now
        self.alerted_tracks=set()
        current={t['track_id']:t['bbox'] for t in tracks if t.get('track_id',-1)>=0}
        jumps={tid for tid,box in current.items() if tid in self.track_boxes and not continuous_box(self.track_boxes[tid],box)}
        current_objects={o['track_id']:o['bbox'] for o in objects if o.get('track_id',-1)>=0}
        object_jumps={oid for oid,box in current_objects.items() if oid in self.object_boxes and not continuous_box(self.object_boxes[oid],box)}
        self.track_boxes=current; self.object_boxes=current_objects
        candidates={}
        def add(key,kind,subject,detail):
            candidates[key]=(kind,str(subject),detail)
        for t in tracks:
            tid=t.get('track_id',-1)
            if tid<0 or tid in jumps:continue
            detail=dict(track_id=tid,person_bbox=t['bbox'],bbox=t['bbox'],
                        attribution='person_track',review_required=True)
            direction=t.get('direction')
            if direction in ('LEFT','RIGHT','UP'):
                add(('head',tid,direction),'HEAD_'+direction,tid,
                    dict(detail,yaw=t.get('yaw'),turn_ratio=t.get('turn_ratio'),head_source=t.get('head_source','yunet'),head_method=t.get('head_method'),head_confidence=t.get('head_confidence'),head_keypoints=t.get('head_keypoints'),description='Quay/ngửa đầu; cần đối chiếu video, chưa kết luận gian lận'))
            if t.get('left_initial_position'):
                add(('motion',tid),'LEFT_INITIAL_POSITION',tid,dict(detail,initial_position=t.get('initial_position'),displacement_px=t.get('motion_pixels'),description='Di chuyển khỏi vị trí ban đầu; cần đối chiếu video'))
            if t.get('body_action') in ('BODY_TURN','BODY_LEAN'):
                add(('body',tid,t['body_action']),t['body_action'],tid,dict(detail,description='Xoay/ngả thân rõ rệt; cần đối chiếu video'))
            pitch=t.get('pitch')
            if self.settings['pitch_enabled'] and direction!='POSE_UNKNOWN' and pitch is not None:
                delta=(pitch-self.settings['pitch_center']+180)%360-180
                if abs(delta)>self.settings['pitch_threshold']:
                    add(('pitch',tid),'HEAD_PITCH',tid,dict(detail,pitch=pitch,description='Góc đầu lệch mốc hiệu chuẩn'))
        for obj in objects:
            oid=obj.get('track_id',-1); cls=obj.get('class_id')
            if oid<0 or oid in object_jumps:continue
            kind={67:'PHONE',73:'BOOK',63:'LAPTOP'}.get(cls)
            if not kind:continue
            if kind=='BOOK' and not self.settings['book_enabled']:continue
            if kind=='LAPTOP' and not self.settings['laptop_enabled']:continue
            x1,y1,x2,y2=obj['bbox'];cx,cy=(x1+x2)/2,(y1+y2)/2
            area=max(1,(x2-x1)*(y2-y1))
            # A single nearby box does not prove ownership. Overlap stays unassigned.
            owners=[]
            for t in tracks:
                a,b,c,d=t['bbox']
                overlap=max(0,min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))/area
                if t.get('track_id',-1)>=0 and a<=cx<=c and b<=cy<=d and overlap>=.8:
                    owners.append(t)
            owner=owners[0] if len(owners)==1 else None
            tid=owner['track_id'] if owner else None
            if tid in jumps:continue
            # Include object ID AND candidate owner: never pool different objects.
            key=('object',kind,oid,tid)
            detail=dict(bbox=obj['bbox'],object_bbox=obj['bbox'],object_track_id=oid,
                        confidence=obj['confidence'],candidate_track_id=tid,
                        person_bbox=owner['bbox'] if owner else None,
                        attribution='near_person_unconfirmed' if owner else 'unassigned',
                        review_required=True,description='Vật thể xuất hiện gần người; chưa xác nhận sử dụng hoặc sở hữu' if owner else 'Phát hiện vật thể; chưa xác định người liên quan')
            add(key,kind,'track:'+str(tid) if owner else 'object:'+str(oid),detail)
        for key in list(self.pending):
            if key not in candidates:del self.pending[key]
        emitted=[]
        for key,(kind,subject,detail) in candidates.items():
            since=self.pending.setdefault(key,now)
            duration=now-since
            hold=self.settings['head_hold_seconds'] if key[0]=='head' else self.settings['hold_seconds']
            if key[0] in ('motion','body'):hold=0.0
            if duration<hold:continue
            # Only head evidence belongs directly to a person. Proximity stays amber.
            if key[0] in ('motion','body','head'):self.alerted_tracks.add(detail['track_id'])
            if now-self.last.get(key,float('-inf'))<self.settings['cooldown_seconds']:continue
            emitted.append(dict(type=kind,subject=subject,duration=round(duration,2),
                                start_time=round(since,3),end_time=round(now,3),**detail))
            self.last[key]=now
        # Limit cooldown state for vanished tracks/objects.
        for key in list(self.last):
            if key not in candidates and now-self.last[key]>self.settings['cooldown_seconds']:
                del self.last[key]
        return emitted
