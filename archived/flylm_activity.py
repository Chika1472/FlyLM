"""Record real FlyLM states and render every neuron in an offline 2D player."""
import base64
import codecs
import gzip
import html
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
import torch

ANNOTATION_URL = (
    'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
    'body-annotations-male-cns-v1.0-minconf-0.5.feather'
)


def neuron_layout(model, graph_dir, cache_dir, annotation_path=None, coordinate_path=None):
    """Join official soma/skeleton representatives by body ID, keeping source provenance."""
    graph_dir, cache_dir = Path(graph_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    body_ids = np.load(graph_dir / 'body_ids.npy', allow_pickle=False)
    if body_ids.shape != (model.n,) or len(np.unique(body_ids)) != model.n:
        raise ValueError('body_ids.npy does not match the model neuron count')
    if coordinate_path is not None:
        with np.load(coordinate_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            if metadata['dataset'] != 'male-cns:v1.0' or metadata['units'] != '8nm_voxels':
                raise ValueError('Coordinate dataset/units do not match MaleCNS v1.0')
            ids, coordinates, sources = saved['body_ids'], saved['xyz'], saved['source']
            if (coordinates.shape != (len(ids), 3) or sources.shape != (len(ids),)
                    or len(np.unique(ids)) != len(ids) or not np.isin(sources, [0, 1, 2, 3]).all()):
                raise ValueError('Invalid coordinate table')
            order = np.argsort(ids)
            positions = np.searchsorted(ids[order], body_ids)
            if (positions >= len(ids)).any() or not np.array_equal(ids[order][positions], body_ids):
                raise ValueError('Coordinate table is missing model body IDs')
            xyz = coordinates[order[positions]].copy()
            coordinate_kind = sources[order[positions]].astype(np.uint8)
            if not np.array_equal(np.isfinite(xyz).all(1), coordinate_kind > 0):
                raise ValueError('Coordinate validity and provenance disagree')
    else:
        import pyarrow.feather as feather

        if annotation_path is not None and not Path(annotation_path).is_file():
            raise FileNotFoundError(f'Annotation file not found: {annotation_path}')
        annotation_path = Path(annotation_path) if annotation_path else cache_dir / 'body-annotations.feather'
        if not annotation_path.exists():
            print('Downloading official soma annotations (about 13 MB)...', flush=True)
            temporary = annotation_path.with_suffix('.partial')
            urllib.request.urlretrieve(ANNOTATION_URL, temporary)
            temporary.replace(annotation_path)
        table = feather.read_table(annotation_path, columns=['bodyId', 'somaLocation'])
        ids = table['bodyId'].to_numpy()
        order = np.argsort(ids)
        sorted_ids = ids[order]
        locations = table['somaLocation'].to_pylist()
        positions = np.searchsorted(sorted_ids, body_ids)
        found = positions < len(sorted_ids)
        found[found] &= sorted_ids[positions[found]] == body_ids[found]
        xyz = np.full((model.n, 3), np.nan, dtype=np.float32)
        for index in np.flatnonzero(found):
            value = locations[int(order[positions[index]])]
            if value is not None and len(value) == 3 and np.isfinite(value).all():
                xyz[index] = value
        coordinate_kind = np.isfinite(xyz).all(1).astype(np.uint8)
    known = np.isfinite(xyz).all(1)
    if not known.any():
        raise ValueError('No coordinates matched. Check body IDs and the coordinate source.')
    roles = np.zeros(model.n, dtype=np.uint8)
    roles[model.inputs.detach().cpu().numpy()] = 1
    roles[model.outputs.detach().cpu().numpy()] = 2
    xy = np.zeros((model.n, 2), dtype=np.float32)
    xy_front = np.full_like(xy, -2)  # Missing positions are drawn once, in the shared tray.
    # Rotate the camera 90 degrees about X: X-Z -> X-Y, not a 2D image rotation.
    # Preserve proportions at the player's nominal 1200 x 560 canvas size.
    aspect = 1200 / 560
    centers, view_width = ((0.25, 0.75), 0.43) if known.all() else ((0.19, 0.53), 0.30)
    for coordinates, vertical_axis, center_x in [(xy, 2, centers[0]), (xy_front, 1, centers[1])]:
        projected = xyz[known][:, [0, vertical_axis]]
        span = np.maximum(np.ptp(projected, axis=0), 1)
        scale = min(view_width * aspect / span[0], 0.78 / span[1])
        centered = projected - (projected.min(0) + projected.max(0)) / 2
        coordinates[known] = centered * [scale / aspect, scale] + [center_x, 0.52]
    # Missing somas often belong to peripheral sensory neurons. Do not invent anatomical positions.
    for role, low, high in [(1, 0.19, 0.39), (0, 0.49, 0.72), (2, 0.82, 0.93)]:
        members = np.flatnonzero(~known & (roles == role))
        if not len(members):
            continue
        columns = max(1, math.ceil(math.sqrt(len(members) * 0.28 / (high - low))))
        rows = math.ceil(len(members) / columns)
        indices = np.arange(len(members))
        xy[members, 0] = 0.76 + 0.21 * ((indices % columns) + 0.5) / columns
        xy[members, 1] = low + (high - low) * ((indices // columns) + 0.5) / rows
    return dict(xy=xy, xy_front=xy_front, roles=roles, body_ids=body_ids, known=known,
                xyz=xyz, coordinate_kind=coordinate_kind,
                coordinate_source=str(coordinate_path) if coordinate_path is not None else ANNOTATION_URL)


@torch.no_grad()
def record_activity(model, runtime, question, max_new_bytes=180, temperature=0.0, max_frames=160):
    """Observe the existing reply() without changing token selection or the dynamics."""
    if max_frames < 4 or max_new_bytes < 1:
        raise ValueError('max_frames >= 4 and max_new_bytes >= 1 required')
    prefix = runtime.inference_inputs(question)
    stride = max(1, math.ceil((len(prefix) + max_new_bytes) / (max_frames - 3)))
    frames = [np.zeros(model.n, dtype=np.uint8)]
    captions = [dict(step=0, phase='ready', input='', output='')]
    input_decoder = codecs.getincrementaldecoder('utf-8')('strict')
    output_decoder = codecs.getincrementaldecoder('utf-8')('strict')
    input_parts, output_parts = [], []
    step, latest = 0, None
    original = model.advance
    had_override = 'advance' in model.__dict__
    prior_override = model.__dict__.get('advance')

    def save_frame(state, phase):
        # s is nonnegative synaptic activity, NOT a spike event. A fixed sqrt display scale
        # retains small signals in uint8; it does not normalize each neuron or each frame.
        signal = state[1, 0]
        if not bool(torch.isfinite(signal).all()):
            raise RuntimeError('Nonfinite neuron activity')
        frames.append(signal.clamp(0, 1).sqrt().mul(255).round().to(torch.uint8).cpu().numpy())
        captions.append(dict(step=step, phase=phase, input=''.join(input_parts), output=''.join(output_parts)))

    def observe(token, *args, **kwargs):
        nonlocal step, latest
        latest = original(token, *args, **kwargs)
        step += 1
        value = int(token[0])
        phase = 'input' if step <= len(prefix) else 'output'
        if value < 256:
            decoder = input_decoder if phase == 'input' else output_decoder
            parts = input_parts if phase == 'input' else output_parts
            parts.append(decoder.decode(bytes([value]), final=False))
        if step % stride == 0 or step == len(prefix):
            save_frame(latest, phase)
        if step % 64 == 0:
            print(f'Recording {phase}: {step} bytes processed', flush=True)
        return latest

    model.advance = observe
    try:
        answer = runtime.reply(model, question, max_new_bytes=max_new_bytes, temperature=temperature)
        save_frame(latest, 'done')
        captions[-1]['output'] = answer
    finally:
        if had_override:
            model.advance = prior_override
        else:
            del model.advance
    return dict(activity=np.stack(frames), captions=captions, question=question,
                answer=answer, stride=stride, brain_steps=model.k)


PLAYER_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlyLM · 뉴런 활동</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#090e19;color:#e3eafa;font:14px/1.55 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{margin:0;font-size:24px;font-weight:650;letter-spacing:-.7px}.eyebrow{color:#7d95bd;font-size:11px;letter-spacing:2px}
.sub{color:#93a5c3;font-size:12px;margin:5px 0 15px}.pill{border:1px solid #34405c;border-radius:20px;padding:5px 13px;color:#b1c4e8}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:#bbcae4}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.stage{position:relative;margin:16px 0 12px;border:1px solid #26334c;border-radius:12px;overflow:hidden;background:radial-gradient(ellipse at 32% 30%,#111d34,#080d17 75%)}
canvas{display:block;width:100%;height:560px}.map-label{position:absolute;top:17px;left:5%;color:#8194b5;font-size:12px;pointer-events:none}
.tray-label{position:absolute;left:75%;color:#a6b8d6;font-size:11px;pointer-events:none}.divider{position:absolute;left:72%;top:9%;bottom:5%;border-left:1px dashed #27354d;pointer-events:none}
#error{position:absolute;inset:25%;background:#182239;color:#ffbfad;padding:24px;display:none}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button,select{background:#16233b;color:#ecf3ff;border:1px solid #344b6f;border-radius:7px;padding:8px 12px;font:inherit;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #71d7ed;outline-offset:3px}button{min-width:85px}
#timeline{flex:1;min-width:140px;accent-color:#74cbe4}.small{font-size:12px;color:#a5b5cf}input[type=range]{accent-color:#79c6eb}
.texts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}.text{min-height:100px;background:#111a2b;border:1px solid #24324c;border-radius:10px;padding:14px}
.text label{font-size:11px;color:#8fa4c8;letter-spacing:1px}.text pre{white-space:pre-wrap;word-break:break-word;font:14px/1.65 system-ui,sans-serif;margin:9px 0 0}
footer{margin-top:14px;color:#7f91b0;font-size:11px}#notice{color:#ffcd86;font-weight:600}
@media(max-width:650px){main{padding:14px}canvas{height:440px}.texts{grid-template-columns:1fr}h1{font-size:20px}.legend{gap:10px}.map-label,.tray-label{font-size:9px}}
</style></head><body><main>
<header><div><div class="eyebrow">FLYLM / ACTIVITY OBSERVER</div><h1>초파리 뉴런의 활동</h1></div><span class="pill" id="phase">불러오는 중</span></header>
<div class="sub" id="summary"></div><div class="sub" id="locationSummary"></div><div id="notice"></div>
<div class="legend"><span><i class="dot" style="background:#5ce6f3"></i>입력 · 감각 뉴런</span><span><i class="dot" style="background:#8399fc"></i>내부 뉴런</span><span><i class="dot" style="background:#ffb674"></i>출력 · 운동 뉴런</span></div>
<div class="stage"><canvas id="brain" aria-label="전체 뉴런의 2D 활동 지도"></canvas>
<div class="map-label" style="left:3%">기존 시점 · X–Z</div><div class="map-label" id="frontLabel" style="left:38%">90° 회전 시점 · X–Y</div><div class="divider" id="viewDivider" style="left:36%"></div><div class="divider" id="trayDivider"></div>
<div class="tray-label" style="top:3%">좌표 없는 뉴런 · 별도 배열</div>
<div class="tray-label" style="top:14%" id="missingInput"></div><div class="tray-label" style="top:44%" id="missingInternal"></div><div class="tray-label" style="top:77%" id="missingOutput"></div><div id="error" role="alert"></div></div>
<div class="controls"><button id="play" disabled>재생</button><input id="timeline" type="range" min="0" max="0" value="0" aria-label="재생 프레임">
<span id="counter" class="small"></span><label class="small">속도 <select id="speed" aria-label="재생 속도"><option value="6">6 fps</option><option value="12" selected>12 fps</option><option value="24">24 fps</option></select></label>
<label class="small">밝기 <input id="gain" aria-label="활동 밝기" type="range" min="0.5" max="8" step="0.25" value="3" style="width:80px"></label></div>
<div class="texts"><div class="text"><label>읽고 있는 입력</label><pre id="inputText"></pre></div><div class="text"><label>생성 중인 답변</label><pre id="outputText"></pre></div></div>
<footer>밝기 = 모델의 시냅스 활동 s (고정 √s 스케일). 실제 스파이크 측정이 아닙니다. 한 점은 한 뉴런입니다.<br>
위치: MaleCNS v1.0 세포체 또는 골격 대표점. 골격 대표점은 세포체 위치가 아닙니다. 두 시점은 같은 뉴런 활동을 동시에 표시합니다.<br>미확인 좌표가 있으면 별도 배열에 표시합니다. 시간은 byte 처리 순서이며 재생 속도는 생물학적 시간이 아닙니다.<br><span id="sampling"></span></footer>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
(async()=>{
const d=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id);
const fail=message=>{ $('error').style.display='block';$('error').textContent=message;throw new Error(message); };
async function unpack(encoded){
 const raw=atob(encoded),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
 if(typeof DecompressionStream==='undefined')fail('이 브라우저는 압축된 활동 데이터를 지원하지 않습니다. 최신 Chrome 또는 Edge에서 HTML을 여세요.');
 return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
const [xyRaw,frontRaw,roleRaw,activityRaw]=await Promise.all([unpack(d.xy),unpack(d.xy_front),unpack(d.roles),unpack(d.activity)]);
const xy=new Float32Array(xyRaw),front=new Float32Array(frontRaw),roles=new Uint8Array(roleRaw),activity=new Uint8Array(activityRaw);
const n=d.neurons,total=d.captions.length,canvas=$('brain');
if(xy.length!==n*2||front.length!==n*2||roles.length!==n||activity.length!==n*total)fail('활동 데이터 크기가 일치하지 않습니다.');
const gl=canvas.getContext('webgl',{alpha:true,antialias:false,preserveDrawingBuffer:true});
if(!gl)fail('WebGL을 사용할 수 없습니다. HTML을 Chrome/Edge에서 열고 하드웨어 가속을 확인하세요.');
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))fail(gl.getShaderInfoLog(s));return s;}
const program=gl.createProgram();
gl.attachShader(program,shader(gl.VERTEX_SHADER,`
 attribute vec2 position;attribute float role;attribute float activity;
 uniform float gain;uniform float pixelRatio;varying vec3 color;varying float light;
 void main(){gl_Position=vec4(position.x*2.0-1.0,1.0-position.y*2.0,0.,1.);
 light=clamp(activity*gain,0.0,1.0);color=role>1.5?vec3(1.,.71,.45):(role>.5?vec3(.36,.90,.95):vec3(.51,.60,.99));
 gl_PointSize=pixelRatio*(1.15+3.5*light);}`));
gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`
 precision mediump float;varying vec3 color;varying float light;
 void main(){vec2 p=gl_PointCoord*2.-1.;float r=dot(p,p);if(r>1.)discard;
 float alpha=exp(-3.*r)*(.05+.64*light);gl_FragColor=vec4(mix(color,vec3(1.),light*.3),alpha);}`));
gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))fail(gl.getProgramInfoLog(program));gl.useProgram(program);
function attribute(name,data,size,type,normalized){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,type,normalized,0,0);return b;}
const positionBuffer=attribute('position',xy,2,gl.FLOAT,false);
attribute('role',roles,1,gl.UNSIGNED_BYTE,false);
const activityBuffer=attribute('activity',activity.subarray(0,n),1,gl.UNSIGNED_BYTE,true);
const gainLocation=gl.getUniformLocation(program,'gain'),ratioLocation=gl.getUniformLocation(program,'pixelRatio');
gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE);gl.clearColor(0,0,0,0);
let frame=0,playing=false,last=0;
function draw(){
 const ratio=Math.min(window.devicePixelRatio||1,2),w=Math.round(canvas.clientWidth*ratio),h=Math.round(canvas.clientHeight*ratio);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);
 gl.uniform1f(gainLocation,Number($('gain').value));gl.uniform1f(ratioLocation,ratio);
 gl.bindBuffer(gl.ARRAY_BUFFER,activityBuffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,activity.subarray(frame*n,(frame+1)*n));
 gl.clear(gl.COLOR_BUFFER_BIT);
 gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer);
 for(const positions of [xy,front]){gl.bufferSubData(gl.ARRAY_BUFFER,0,positions);gl.drawArrays(gl.POINTS,0,n);}
 const c=d.captions[frame];$('phase').textContent=({ready:'준비',input:'입력 읽는 중',output:'답변 생성 중',done:'완료'})[c.phase];
 $('inputText').textContent=c.input||'입력을 기다리는 중';$('outputText').textContent=c.output||(c.phase==='done'?'(빈 답변)':'…');
 $('counter').textContent=`${frame+1} / ${total} · ${c.step} byte`;$('timeline').value=frame;
}
function pause(){playing=false;$('play').textContent='재생';}
$('play').onclick=()=>{if(playing){pause();return;}if(frame===total-1)frame=0;playing=true;last=performance.now();$('play').textContent='일시정지';draw();};
$('timeline').max=total-1;$('timeline').oninput=()=>{pause();frame=Number($('timeline').value);draw();};
$('gain').oninput=draw;new ResizeObserver(draw).observe(canvas);
function tick(now){if(playing&&now-last>=1000/Number($('speed').value)){frame=Math.min(total-1,frame+1);last=now;draw();if(frame===total-1)pause();}requestAnimationFrame(tick);}
$('summary').textContent=`전체 ${n.toLocaleString()}개 · 위치 확인 ${d.known.toLocaleString()}개 · 미확인 ${(n-d.known).toLocaleString()}개`;
$('locationSummary').textContent=`세포체 ${d.location_counts[1].toLocaleString()} · 골격 대표점 ${d.location_counts[2].toLocaleString()} · 기타 주석 위치 ${d.location_counts[3].toLocaleString()}`;
if(d.known===n){$('frontLabel').style.left='53%';$('viewDivider').style.left='50%';$('trayDivider').style.display='none';document.querySelectorAll('.tray-label').forEach(e=>e.style.display='none');}
$('missingInput').textContent=`감각 · ${d.missing[1].toLocaleString()}개`;$('missingInternal').textContent=`내부 · ${d.missing[0].toLocaleString()}개`;$('missingOutput').textContent=`운동 · ${d.missing[2].toLocaleString()}개`;
$('sampling').textContent=`${d.stride} byte 간격으로 기록 · byte당 ${d.brain_steps} simulation ticks · 추론 후 재생`;
$('notice').textContent=d.notice||'';$('play').disabled=false;draw();requestAnimationFrame(tick);
})().catch(e=>{const el=document.getElementById('error');el.style.display='block';el.textContent=e.message;console.error(e);});
</script></body></html>'''


def activity_html(trace, layout, notice=''):
    def pack(array):
        return base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(), compresslevel=3)).decode('ascii')

    n = len(layout['roles'])
    if trace['activity'].shape != (len(trace['captions']), n):
        raise ValueError('Activity shape does not match neuron layout')
    payload = dict(neurons=n, known=int(layout['known'].sum()),
                   location_counts=[int((layout['coordinate_kind'] == code).sum()) for code in range(4)],
                   missing=[int((~layout['known'] & (layout['roles'] == role)).sum()) for role in range(3)],
                   xy=pack(layout['xy'].astype('<f4')), xy_front=pack(layout['xy_front'].astype('<f4')),
                   roles=pack(layout['roles'].astype('u1')),
                   activity=pack(trace['activity'].astype('u1')), captions=trace['captions'],
                   stride=trace['stride'], brain_steps=trace['brain_steps'], notice=notice)
    # User text stays JSON/textContent, never executable HTML.
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    return PLAYER_HTML.replace('__PAYLOAD__', data)


def show_activity(trace, layout, output_path):
    from IPython.display import HTML, display

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = activity_html(trace, layout)
    output_path.write_text(document, encoding='utf-8')
    print('Offline activity player:', output_path)
    print('Answer:', trace['answer'])
    display(HTML('<iframe sandbox="allow-scripts" style="width:100%;height:1040px;border:0" '
                 'title="FlyLM neuron activity" srcdoc="' + html.escape(document, quote=True) + '"></iframe>'))
    return output_path
