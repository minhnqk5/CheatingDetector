"""SQLite: mỗi thao tác một kết nối, transaction; không lưu dữ liệu mặt ra API."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

def utc():
    return datetime.now(timezone.utc).isoformat()

class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS students(id TEXT PRIMARY KEY, name TEXT NOT NULL, seat TEXT UNIQUE NOT NULL, embedding TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS seats(name TEXT PRIMARY KEY, box TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, name TEXT, started TEXT, ended TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, session_id INTEGER, created TEXT, media_time REAL, type TEXT, subject TEXT, payload TEXT, image TEXT, state TEXT DEFAULT 'pending', note TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, created TEXT, action TEXT, payload TEXT);
            ''')
    def connect(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        return c
    def audit(self, action, payload):
        with self.connect() as c:
            c.execute('INSERT INTO audit(created,action,payload) VALUES(?,?,?)', (utc(), action, json.dumps(payload, ensure_ascii=False)))
    def start(self, name):
        with self.connect() as c:
            return c.execute('INSERT INTO sessions(name,started) VALUES(?,?)',(name,utc())).lastrowid
    def end(self, sid):
        with self.connect() as c:
            c.execute('UPDATE sessions SET ended=? WHERE id=?',(utc(),sid))
    def sessions(self):
        with self.connect() as c:
            return [dict(r) for r in c.execute('SELECT * FROM sessions ORDER BY id DESC')]
    def event(self, sid, media, event, image):
        with self.connect() as c:
            return c.execute('INSERT INTO events(session_id,created,media_time,type,subject,payload,image) VALUES(?,?,?,?,?,?,?)', (sid,utc(),media,event['type'],str(event['subject']),json.dumps(event,ensure_ascii=False),image)).lastrowid
    def events(self, sid=None, limit=500):
        with self.connect() as c:
            rows=c.execute('SELECT * FROM events WHERE (? IS NULL OR session_id=?) ORDER BY id DESC LIMIT ?', (sid,sid,limit))
            return [dict(r, payload=json.loads(r['payload'])) for r in rows]
    def review(self, eid, state, note):
        if state not in ('confirmed','dismissed','pending'):
            raise ValueError('Trạng thái không hợp lệ')
        with self.connect() as c:
            if c.execute('UPDATE events SET state=?,note=? WHERE id=?',(state,note,eid)).rowcount!=1:
                raise ValueError('Không tìm thấy cảnh báo')
            c.execute('INSERT INTO audit(created,action,payload) VALUES(?,?,?)',(utc(),'review',json.dumps(dict(event_id=eid,state=state,note=note),ensure_ascii=False)))

    def settings(self):
        with self.connect() as c:
            return {r['key']:json.loads(r['value']) for r in c.execute('SELECT * FROM settings')}
    def save_settings(self, values):
        with self.connect() as c:
            c.executemany('INSERT OR REPLACE INTO settings VALUES(?,?)',
                          [(k,json.dumps(v)) for k,v in values.items()])

    def delete_events(self, eid=None):
        """Remove rows atomically; return images no longer referenced by any event."""
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            rows=c.execute('SELECT id,image FROM events WHERE (? IS NULL OR id=?)',(eid,eid)).fetchall()
            if eid is not None and not rows:
                raise ValueError('Không tìm thấy cảnh báo')
            c.execute('DELETE FROM events WHERE (? IS NULL OR id=?)',(eid,eid))
            images=[]
            for name in {r['image'] for r in rows if r['image']}:
                if c.execute('SELECT 1 FROM events WHERE image=? LIMIT 1',(name,)).fetchone() is None:
                    images.append(name)
            c.execute('INSERT INTO audit(created,action,payload) VALUES(?,?,?)',
                      (utc(),'delete_events',json.dumps({'event_id':eid,'count':len(rows)})))
            return len(rows),images
