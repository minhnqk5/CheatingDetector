'use strict';

const $ = id => document.getElementById(id);

let csrf = '';
let enabled = false;
let busy = false;
let streamMode = '';
let lastEvents = '';

const labels = {
    HEAD_LEFT: 'Phát hiện vi phạm',
    HEAD_RIGHT: 'Phát hiện vi phạm',
    HEAD_PITCH: 'Phát hiện vi phạm',
    LOOKING_LEFT_LONG: 'Phát hiện vi phạm',
    LOOKING_RIGHT_LONG: 'Phát hiện vi phạm',
    LOOKING_BACK_LONG: 'Phát hiện vi phạm',
    POSE_UNKNOWN_LONG: 'Phát hiện vi phạm',
    PHONE: 'Phát hiện vi phạm',
    PHONE_DETECTED: 'Phát hiện vi phạm',
    BOOK: 'Phát hiện vi phạm',
    BOOK_DETECTED: 'Phát hiện vi phạm',
    LAPTOP: 'Phát hiện vi phạm',
    LAPTOP_DETECTED: 'Phát hiện vi phạm'
};

function node(tag, text, cls) {
    const n = document.createElement(tag);

    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;

    return n;
}

function notice(text) {
    $('notice').textContent = text;
    $('notice').hidden = !text;
    $('notice').style.display = text ? 'block' : 'none';
}

async function api(path, method = 'GET', body) {
    const headers = {};
    let data = body;

    if (method !== 'GET') {
        headers['X-CSRF-Token'] = csrf;
    }

    if (body && !(body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
        data = JSON.stringify(body);
    }

    const r = await fetch(path, {
        method,
        headers,
        body: data
    });

    const result = await r.json();

    if (!r.ok) {
        throw Error(result.error || 'Yêu cầu thất bại');
    }

    return result;
}

function stream(mode) {
    if (
        streamMode !== mode ||
        !$('stream').getAttribute('src')
    ) {
        $('stream').src =
            '/video_feed?view=' + mode + '&t=' + Date.now();

        streamMode = mode;
    }
}

async function mutate(fn) {
    if (busy) return;

    busy = true;

    $('toggle').disabled = true;
    $('open').disabled = true;
    $('upload').disabled = true;

    try {
        await fn();
        notice('');
    } catch (e) {
        notice(e.message);
    } finally {
        busy = false;

        $('open').disabled = false;
        $('upload').disabled = false;

        await status().catch(e => notice(e.message));
    }
}

// Bật / tắt AI
$('toggle').onclick = () => mutate(
    () => api('/api/analysis', 'POST', {
        enabled: !enabled
    })
);

// Xóa toàn bộ lịch sử
$('delete-all').onclick = async () => {
    if (!confirm(
        'Xóa toàn bộ lịch sử cảnh báo và ảnh bằng chứng không còn được dùng?'
    )) {
        return;
    }

    try {
        const r = await api('/api/events', 'DELETE');

        lastEvents = '';

        notice(`Đã xóa ${r.deleted} cảnh báo.`);

        await eventsRefresh();

    } catch (e) {
        notice(e.message);
    }
};

// Tải video
$('upload').onchange = () => {
    const file = $('upload').files[0];

    if (!file) return;

    mutate(async () => {
        notice('Đang tải video…');

        const form = new FormData();
        form.append('video', file);

        await api('/api/upload', 'POST', form);

        $('filename').textContent = file.name;

        streamMode = '';
    }).finally(() => {
        $('upload').value = '';
    });
};

// Mở webcam (0 = Webcam máy tính, 1 = Camera USB)
$('open').onclick = () => mutate(async () => {
    const source = $('source').value.trim();

    if (!/^[01]$/.test(source)) {
        throw Error(
            'Chỉ nhập 1 (Webcam) hoặc 0 (Camera USB)'
        );
    }

    await api('/api/source', 'POST', {
        mode: 'camera',
        source: Number(source)
    });

    $('filename').textContent =
        source === '0'
            ? 'Webcam USB'
            : 'Webcam máy tính';

    streamMode = '';
});

// Chuyển chế độ hiển thị
$('view').onchange = () => {
    streamMode = '';
};

// Khi ảnh video tải xong
$('stream').onload = () => {
    $('placeholder').hidden = true;
};

// Cập nhật trạng thái
async function status() {
    const s = await api('/api/status');

    enabled = s.analysis_enabled;

    $('toggle').textContent = enabled ? 'Tắt AI' : 'Bật AI';

    $('toggle').disabled =
        busy || s.mode === 'idle' || s.finished;

    $('analysis-state').textContent = s.finished
        ? 'Đã phân tích hết video'
        : enabled
            ? 'Đang phân tích'
            : 'Đang tắt';

    $('connection').textContent = s.error ||
        (s.ready
            ? '● Nguồn sẵn sàng'
            : '○ Chưa mở nguồn');

    $('persons').textContent = s.persons;

    $('fps').textContent = enabled
        ? s.ai_fps
        : '—';

    $('samples').textContent = s.sampled_frames;

    $('media').textContent = s.media_time == null
        ? '—'
        : s.media_time.toFixed(2) + ' s';

    $('view').hidden =
        s.mode === 'video' || s.mode === 'idle';

    const cadence = 'AI: 1/' + s.frame_stride + ' frame';

    $('details').textContent =
        cadence +
        ' · YOLO ' + (s.timings?.yolo_ms ?? '—') + ' ms' +
        ' · Hướng đầu ' + (s.timings?.face_ms ?? '—') + ' ms' +
        ' · Frame bỏ qua ' + s.skipped_frames +
        (s.capture_fps != null
            ? ' · Camera ' + s.capture_fps + ' FPS'
            : '');

    if (s.mode !== 'idle') {
        stream(
            s.mode === 'video'
                ? 'analysis'
                : $('view').value
        );

        $('placeholder').hidden = true;
    }

    if (s.error) {
        notice(s.error);
    }
}

// Hiển thị lịch sử cảnh báo
function renderEvents(events) {
    const key = JSON.stringify(events);

    if (
        key === lastEvents ||
        document.activeElement?.closest('.event')
    ) {
        return;
    }

    lastEvents = key;

    const root = $('events');
    root.replaceChildren();

    const visible = events.filter(e => labels[e.type]);

    if (!visible.length) {
        root.append(node('p', 'Chưa có cảnh báo.'));
        return;
    }

    for (const e of visible) {
        const card = node('article', undefined, 'event');

        card.append(
            node('b', labels[e.type]),

            node(
                'small',
                new Date(e.created).toLocaleString('vi-VN')
            )
        );

        if (e.image) {
            const a = node('a');

            a.href =
                '/evidence/' + encodeURIComponent(e.image);

            a.target = '_blank';
            a.rel = 'noopener';

            const img = node('img');

            img.src = a.href;
            img.loading = 'lazy';
            img.alt = 'Bằng chứng cảnh báo';

            a.append(img);
            card.append(a);
        }

        if (e.media_time != null) {
            card.append(
                node(
                    'small',
                    'Video: ' + e.media_time.toFixed(2) + ' s'
                )
            );
        }

        card.append(
            node(
                'p',
                e.payload?.description ||
                ('Duy trì ' + (e.payload?.duration ?? 0) + ' giây')
            )
        );

        const del = node('button', 'Xóa');

        del.className = 'danger';

        del.onclick = async () => {
            try {
                const r = await api(
                    '/api/events/' + e.id,
                    'DELETE'
                );

                lastEvents = '';

                notice(
                    'Đã xóa cảnh báo và ' +
                    r.deleted +
                    ' ảnh bằng chứng không còn được dùng.'
                );

                await eventsRefresh();

            } catch (err) {
                notice(err.message);
            }
        };

        card.append(del);
        root.append(card);
    }
}

async function eventsRefresh() {
    renderEvents(await api('/api/events'));
}

// Kiểm tra trạng thái định kỳ
async function poll() {
    try {
        await status();

    } catch (e) {
        $('connection').textContent =
            'Mất kết nối: ' + e.message;
    }

    setTimeout(poll, 800);
}

// Cập nhật cảnh báo định kỳ
async function slowPoll() {
    try {
        await eventsRefresh();

    } catch (e) {
        notice(e.message);
    }

    setTimeout(slowPoll, 4000);
}

// Khởi động ứng dụng
(async () => {
    try {
        csrf = (await api('/api/session')).csrf;

        $('source').value = '0';
        $('source').placeholder = '1: Webcam | 0: USB';

        await status();

        poll();
        slowPoll();

    } catch (e) {
        notice(e.message);
    }
})();