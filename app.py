import atexit
import csv
import io
import math
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from flask import Flask, Response, jsonify, render_template, request, session, send_from_directory
from werkzeug.exceptions import HTTPException
from store import Store
from engine import Engine

ROOT=Path(__file__).resolve().parent

def create_app(data_dir=None, start_worker=True):
    data=Path(data_dir or os.environ.get('EXAM_DATA',ROOT/'data'))
    data.mkdir(parents=True,exist_ok=True)
    app=Flask(__name__)
    app.config.update(SECRET_KEY=secrets.token_hex(32),MAX_CONTENT_LENGTH=512*1024*1024,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict')
    store=Store(data/'exam.sqlite3')
    # Ca dang dở do tiến trình dừng không được tự tiếp tục bằng trạng thái tracking mới.
    for s in store.sessions():
        if not s['ended']:
            store.end(s['id']); store.audit('session_recovered_after_restart',{'id':s['id']})
    engine=Engine(store,data/'evidence')
    app.extensions.update(exam_store=store,exam_engine=engine)
    if start_worker:
        engine.start_worker(); atexit.register(engine.close)
    @app.before_request
    def protect():
        # No login. Automatic browser session only protects mutation requests from CSRF.
        if request.method not in ('GET','HEAD','OPTIONS') and not secrets.compare_digest(request.headers.get('X-CSRF-Token',''),session.get('csrf','!')):
            return jsonify(error='Phiên yêu cầu không hợp lệ; tải lại trang'),403
    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['X-Frame-Options']='DENY'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        return response
    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc,HTTPException):
            return jsonify(error=exc.description),exc.code
        if isinstance(exc,(ValueError,TypeError,KeyError)):
            return jsonify(error=str(exc)),400
        app.logger.exception('Request failed')
        return jsonify(error='Lỗi xử lý. Xem terminal để biết chi tiết.'),500
    @app.get('/')
    def home():
        return render_template('dashboard.html')
    @app.get('/api/session')
    def browser_session():
        if 'csrf' not in session:
            session['csrf']=secrets.token_urlsafe(24)
        return jsonify(csrf=session['csrf'])
    @app.get('/api/status')
    def status():
        return jsonify(engine.snapshot())
    @app.get('/api/pose-debug')
    def pose_debug():
        state=engine.snapshot()
        fields=('track_id','bbox','direction','raw_direction','head_reason','head_method',
                'head_confidence','turn_ratio','pose_source','head_keypoints',
                'stable_samples','pose_retry_error','alert_confirmed')
        return jsonify(tracks=[{k:t.get(k) for k in fields} for t in state.get('tracks',[])],
                       mode=state.get('mode'),error=state.get('error'))
    @app.get('/video_feed')
    def feed():
        analysis=request.args.get('view')=='analysis'
        def frames():
            last=None
            while engine.running:
                jpg=engine.wait_preview(last,analysis)
                if jpg and jpg is not last:
                    last=jpg
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg+b'\r\n'
                if not jpg:
                    time.sleep(0.01)
        return Response(frames(),mimetype='multipart/x-mixed-replace; boundary=frame',headers={'X-Accel-Buffering':'no'})
    @app.get('/api/frame')
    def frame():
        with engine.state_lock:
            if not engine.jpg:
                return jsonify(error='Chưa có ảnh'),404
            return Response(engine.jpg,mimetype='image/jpeg')
    @app.post('/api/source')
    def source():
        body=request.get_json() or {}
        mode=body.get('mode')
        if mode=='camera':
            value=int(body.get('source',0))
            if not 0<=value<=20:
                raise ValueError('Chỉ số camera phải từ 0 đến 20')
        elif mode=='ip':
            value=str(body.get('source','')).strip()
            parsed=urlsplit(value)
            if parsed.scheme not in ('http','https','rtsp') or not parsed.hostname:
                raise ValueError('URL camera phải là http, https hoặc rtsp')
        else:
            raise ValueError('Nguồn không hợp lệ')
        engine.source(value,mode); return jsonify(ok=True)
    @app.post('/api/upload')
    def upload():
        f=request.files.get('video')
        if not f or Path(f.filename).suffix.lower() not in ('.mp4','.avi','.mov','.mkv'):
            raise ValueError('Chọn video mp4, avi, mov hoặc mkv')
        target=data/'uploads'/(uuid4().hex+Path(f.filename).suffix.lower())
        target.parent.mkdir(exist_ok=True); f.save(target)
        try:
            engine.source(str(target),'video')
        except Exception:
            target.unlink(missing_ok=True); raise
        return jsonify(ok=True)
    @app.post('/api/settings')
    def settings():
        body=request.get_json() or {}
        ranges={'head_hold_seconds':(0,60),'hold_seconds':(1,60),'absence_seconds':(3,300),'cooldown_seconds':(3,600),'max_gap_seconds':(0.2,10),'pitch_center':(-180,180),'pitch_threshold':(10,90)}
        changes={}
        for key,value in body.items():
            if key in ('pitch_enabled','book_enabled','laptop_enabled'):
                if not isinstance(value,bool):
                    raise ValueError(key+' phải là boolean')
                changes[key]=value
            elif key in ranges:
                value=float(value); low,high=ranges[key]
                if not math.isfinite(value) or not low<=value<=high:
                    raise ValueError('Ngưỡng ngoài phạm vi: '+key)
                changes[key]=value
            else:
                raise ValueError('Cấu hình không tồn tại: '+key)
        with engine.lock:
            if engine.analysis_enabled:
                raise ValueError('Tắt AI trước khi đổi ngưỡng')
            store.save_settings(changes); engine.rules.settings.update(changes); engine.rules.reset(); store.audit('settings',changes)
        return jsonify(ok=True)
    @app.post('/api/analysis')
    def analysis():
        enabled=(request.get_json() or {}).get('enabled')
        if not isinstance(enabled,bool):
            raise ValueError('enabled phải là boolean')
        engine.set_analysis(enabled)
        return jsonify(ok=True,analysis_enabled=engine.analysis_enabled)
    @app.get('/api/events')
    def events():
        return jsonify(store.events(request.args.get('session_id',type=int)))
    @app.delete('/api/events')
    @app.delete('/api/events/<int:eid>')
    def delete_events(eid=None):
        # Prevent racing AI evidence creation/deletion in this process.
        with engine.lock:
            count,images=store.delete_events(eid)
            cleanup_failed=0
            root=(data/'evidence').resolve()
            for name in images:
                path=root/name
                if path.name!=name or path.is_symlink() or path.resolve().parent!=root:
                    cleanup_failed+=1
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    cleanup_failed+=1
                    app.logger.warning('Could not remove evidence file %s',name)
        return jsonify(ok=True,deleted=count,cleanup_failed=cleanup_failed)
    @app.post('/api/events/<int:eid>/review')
    def review(eid):
        body=request.get_json() or {}; note=str(body.get('note',''))[:2000]
        store.review(eid,body.get('state'),note); return jsonify(ok=True)
    @app.get('/evidence/<path:name>')
    def evidence(name):
        return send_from_directory(data/'evidence',name)
    @app.get('/api/export')
    def export():
        rows=store.events(request.args.get('session_id',type=int),limit=1000000)
        out=io.StringIO(); writer=csv.writer(out)
        writer.writerow(['id','session_id','UTC','video_seconds','type','subject','state','note','evidence','details'])
        def safe(value):
            text='' if value is None else str(value)
            return "'"+text if text[:1] in '=+-@\t\r' else text
        for e in rows:
            writer.writerow([safe(e[k]) for k in ('id','session_id','created','media_time','type','subject','state','note','image','payload')])
        return Response('\ufeff'+out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=exam_events.csv'})
    return app

if __name__=='__main__':
    create_app().run(host=os.environ.get('EXAM_HOST','127.0.0.1'),port=5000,threaded=True,debug=False,use_reloader=False)
