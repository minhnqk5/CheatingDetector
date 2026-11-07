"""Read latest frames, encode preview on its own thread, expose camera stats."""
from motion import person_visual
import copy
import threading
import time
import cv2


class LatestCamera:
    def __init__(self, cap, preview_fps=30, preview_width=960, jpeg_quality=70, overlay_max_age=.7):
        self.cap=cap
        self.preview_fps=preview_fps
        self.preview_width=preview_width
        self.jpeg_quality=max(40,min(95,int(jpeg_quality)))
        self.overlay_max_age=overlay_max_age
        self.overlay=None
        self.overlay_version=0
        self.analysis_jpg=None
        self.preview_captured=None
        self.preview_fps_actual=0.0
        self.last_encoded_at=None
        self.preview_sequence=0
        self.lock=threading.Lock()
        self.changed=threading.Condition(self.lock)
        self.stop=threading.Event()
        self.packet=None
        self.jpg=None
        self.error=''
        self.fps=0.0
        self.read_ms=0.0
        self.preview_ms=0.0
        self.thread=threading.Thread(target=self.run,daemon=True,name='CameraLatestFrame')
        self.encoder=threading.Thread(target=self.encode,daemon=True,name='CameraPreviewJPEG')
        self.thread.start()
        self.encoder.start()

    def publish(self,sequence,timestamp,frame,read_ms=0.0):
        with self.changed:
            self.packet=(sequence,timestamp,frame)
            self.read_ms=read_ms
            self.changed.notify_all()

    def latest(self):
        with self.lock:
            return self.packet,self.error,self.fps

    def metrics(self):
        with self.lock:
            return {'camera_read_ms':round(self.read_ms,1),
                    'preview_encode_ms':round(self.preview_ms,1),
                    'preview_fps':round(self.preview_fps_actual,1),
                    'preview_age_ms':round((time.monotonic()-self.preview_captured)*1000,1) if self.preview_captured else None,
                    'capture_width':self.packet[2].shape[1] if self.packet else None,
                    'capture_height':self.packet[2].shape[0] if self.packet else None}

    def wait_preview(self,last=None,timeout=.5,analysis=False):
        with self.changed:
            self.changed.wait_for(lambda:(self.analysis_jpg if analysis else self.jpg) is not None and (self.analysis_jpg if analysis else self.jpg) is not last
                                  or bool(self.error) or self.stop.is_set(),timeout=timeout)
            if self.error or not self.packet or time.monotonic()-self.packet[1]>2:
                return None
            return (self.analysis_jpg or self.jpg) if analysis else self.jpg

    def preview(self,analysis=False):
        with self.lock:
            if self.error or not self.packet or time.monotonic()-self.packet[1]>2:
                return None
            return (self.analysis_jpg or self.jpg) if analysis else self.jpg

    def run(self):
        previous=None
        sequence=0
        try:
            while not self.stop.is_set():
                started=time.perf_counter()
                ok,frame=self.cap.read()
                read_ms=(time.perf_counter()-started)*1000
                if not ok:
                    raise RuntimeError('Mất tín hiệu camera. Kiểm tra webcam và mở lại nguồn.')
                now=time.monotonic();sequence+=1
                with self.lock:
                    if previous is not None:
                        instant=1/max(now-previous,.001)
                        self.fps=instant if not self.fps else self.fps*.9+instant*.1
                previous=now
                self.publish(sequence,now,frame,read_ms)
        except Exception as exc:
            with self.changed:
                self.error=str(exc);self.packet=None;self.jpg=None;self.changed.notify_all()
        finally:
            self.cap.release()

    def set_overlay(self,tracks,objects,captured,shape):
        # Never render the mutable Monitor state from the encoder thread.
        value=copy.deepcopy((tracks,objects,captured,shape))
        with self.changed:
            self.overlay=value
            self.overlay_version+=1

    def clear_overlay(self):
        with self.changed:
            self.overlay=None
            self.overlay_version+=1
            self.analysis_jpg=self.jpg
            self.changed.notify_all()

    def render_overlay(self,frame,overlay):
        output=frame.copy()
        tracks,objects,captured,shape=overlay
        age=max(0,time.monotonic()-captured)
        if age<=self.overlay_max_age:
            sx=output.shape[1]/shape[1];sy=output.shape[0]/shape[0]
            for person in tracks:
                x1,y1,x2,y2=person['bbox']
                a,b,c,d=int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                color,head_label,body_label=person_visual(person)
                cv2.rectangle(output,(a,b),(c,d),color,2)
                cv2.putText(output,f"ID {person['track_id']} | {head_label} | {body_label}",
                    (a,max(16,b-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            for obj in objects:
                x1,y1,x2,y2=obj['bbox']
                a,b,c,d=int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                color=(0,0,255)
                cv2.rectangle(output,(a,b),(c,d),color,1)
                cv2.putText(output,f"{obj['class_name']} {obj['confidence']:.2f}",
                    (a,max(16,b-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        text=f'LIVE | AI result age: {age:.1f}s'
        if age>self.overlay_max_age:text+=' | waiting for fresh boxes'
        cv2.putText(output,text,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        return output

    def encode(self):
        last=0
        interval=1/max(self.preview_fps,1)
        next_frame_at=0.0
        while not self.stop.is_set():
            with self.changed:
                self.changed.wait_for(lambda:(self.packet is not None and self.packet[0]!=last)
                                      or bool(self.error) or self.stop.is_set(),timeout=.5)
                if self.stop.is_set() or self.error:return
            delay=next_frame_at-time.monotonic()
            if delay>0 and self.stop.wait(delay):return
            # Re-read AFTER waiting. No queue of old frames.
            with self.lock:
                packet=self.packet;overlay=self.overlay;version=self.overlay_version
            if packet is None or packet[0]==last:continue
            last,captured,frame=packet
            started=time.perf_counter()
            try:
                height,width=frame.shape[:2]
                if width>self.preview_width:
                    frame=cv2.resize(frame,(self.preview_width,round(height*self.preview_width/width)),interpolation=cv2.INTER_AREA)
                options=[cv2.IMWRITE_JPEG_QUALITY,self.jpeg_quality]
                ok,encoded=cv2.imencode('.jpg',frame,options)
                if not ok:raise RuntimeError('Không mã hóa được ảnh camera')
                raw=encoded.tobytes();annotated=raw
                if overlay is not None:
                    ok,data=cv2.imencode('.jpg',self.render_overlay(frame,overlay),options)
                    if not ok:raise RuntimeError('Không mã hóa được ảnh có nhãn')
                    annotated=data.tobytes()
                now=time.monotonic()
                with self.changed:
                    # A newer capture arriving during JPEG encode must NOT discard
                    # this completed JPEG (the old condition could starve preview).
                    if self.stop.is_set() or self.error:return
                    self.jpg=raw
                    self.analysis_jpg=annotated if version==self.overlay_version else raw
                    self.preview_captured=captured;self.preview_sequence=last
                    self.preview_ms=(time.perf_counter()-started)*1000
                    if self.last_encoded_at is not None:
                        fps=1/max(now-self.last_encoded_at,.001)
                        self.preview_fps_actual=fps if not self.preview_fps_actual else .8*self.preview_fps_actual+.2*fps
                    self.last_encoded_at=now
                    self.changed.notify_all()
            except Exception as exc:
                with self.changed:
                    self.error='Lỗi tạo ảnh xem trước: '+str(exc)
                    self.jpg=None;self.analysis_jpg=None;self.changed.notify_all()
                self.stop.set();return
            # Encode time is inside the FPS budget, not added to it.
            next_frame_at=max(next_frame_at+interval,time.monotonic())

    def close(self):
        self.stop.set()
        with self.changed:self.changed.notify_all()
        self.thread.join(timeout=6)
        self.encoder.join(timeout=2)
