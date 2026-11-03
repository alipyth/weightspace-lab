const $ = (id) => document.getElementById(id);
const state = { sid: localStorage.getItem('wsl_session') || '', summary: null, tensors: [], selectedTensor: '', selectedSymptom: 'تکرار و حلقه‌زدن', mode: 'ai', proposal: null, preview: null, lastEval: null };

const savedTheme = localStorage.getItem('wsl_theme') || 'dark';
document.body.dataset.theme = savedTheme;
function updateThemeButton(){const b=$('themeToggle');if(b){b.textContent=document.body.dataset.theme==='light'?'☾':'☀';b.title=document.body.dataset.theme==='light'?'رفتن به پوسته تیره':'رفتن به پوسته روشن'}}
updateThemeButton();
$('themeToggle')?.addEventListener('click',()=>{const next=document.body.dataset.theme==='light'?'dark':'light';document.body.dataset.theme=next;localStorage.setItem('wsl_theme',next);updateThemeButton()});

const fa = (v, digits=2) => Number(v ?? 0).toLocaleString('fa-IR', {maximumFractionDigits:digits});
const pct = (v) => `${fa((Number(v)||0)*100,1)}٪`;
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtBytes = (n) => n >= 1024**3 ? `${fa(n/1024**3,2)} گیگابایت` : n >= 1024**2 ? `${fa(n/1024**2,1)} مگابایت` : n >= 1024 ? `${fa(n/1024,1)} کیلوبایت` : `${fa(n,0)} بایت`;

async function api(path, options={}) {
  const headers = new Headers(options.headers || {});
  if (state.sid) headers.set('X-Session-ID', state.sid);
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type','application/json');
  const res = await fetch(path, {...options, headers});
  if (!res.ok) {
    let detail = `خطای ${res.status}`;
    try { const body = await res.json(); detail = body.detail || detail; } catch {}
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res;
}
const post = (path, data={}) => api(path,{method:'POST',body:JSON.stringify(data)});
function toast(message, type='success') {
  const el = document.createElement('div'); el.className=`toast ${type}`; el.textContent=message; $('toastContainer').appendChild(el);
  setTimeout(()=>el.remove(),5200);
}
function busy(show,title='در حال انجام عملیات',text='این فرایند ممکن است بسته به اندازه مدل زمان‌بر باشد.') {
  $('busyTitle').textContent=title; $('busyText').textContent=text; $('busyOverlay').classList.toggle('hidden',!show);
}
async function runBusy(fn,title,text) { busy(true,title,text); try { return await fn(); } finally { busy(false); } }
function setPage(page) {
  document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===`page-${page}`));
  document.querySelectorAll('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.page===page));
  const active=document.querySelector(`.nav-item[data-page="${page}"]`);
  $('breadcrumbTitle').textContent=active?.querySelector('span')?.textContent || 'فضای کار';
  $('sidebar').classList.remove('open'); window.scrollTo({top:0,behavior:'smooth'});
  if(page==='tensors' && state.summary) loadTensors();
  if(page==='editor' && state.summary) fillTensorSelect();
  if(page==='evaluation' && state.sid) loadExperiments();
  if(page==='editor' && state.sid) loadHistory();
}
document.querySelectorAll('.nav-item').forEach(b=>b.addEventListener('click',()=>setPage(b.dataset.page)));
document.querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>setPage(b.dataset.go)));
$('mobileMenu').addEventListener('click',()=>$('sidebar').classList.toggle('open'));

function renderModelProfile(p){
  if(!p){$('modelProfileContent').innerHTML='<div class="empty-mini">مشخصات خودکار مدل در دسترس نیست.</div>';return}
  const cfg=p.config||{};
  const facts=[['معماری',p.family||'—'],['کار اصلی',p.task||'—'],['بلوک‌ها',p.layer_count??'—'],['ابعاد پنهان',cfg.d_model??'—'],['واژگان',cfg.vocab_size??'—'],['زمینه',cfg.max_seq_len??'—']];
  $('modelProfileContent').innerHTML=`<div class="profile-purpose"><span class="profile-chip">${esc(p.family||'مدل شناسایی‌شده')}</span><b>${esc(p.purpose||'هدف مدل هنوز مشخص نیست')}</b><small>اطمینان تشخیص معماری: ${pct(p.confidence||0)} — این مقدار اطمینان به شناسایی نوع مدل است، نه کیفیت آن.</small></div><div class="profile-facts">${facts.map(([k,v])=>`<div><small>${esc(k)}</small><b>${esc(String(v))}</b></div>`).join('')}</div><div class="profile-evidence"><b>از کجا تشخیص داده شد؟</b><ul>${(p.evidence||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div><p class="note">${esc(p.confidence_note||'هدف از معماری و metadata استنباط شده است.')}</p>`;
}
function renderSummary(s) {
  state.summary=s;
  $('weightDiffPanel')?.classList.add('hidden');
  renderModelProfile(s.model_profile);
  $('kpiParams').textContent=s.parameter_count ? (s.parameter_count/1e6>=1 ? `${fa(s.parameter_count/1e6,2)}M` : fa(s.parameter_count,0)) : '—';
  $('kpiTensors').textContent=fa(s.tensor_count||0,0); $('kpiLayers').textContent=s.layers ?? '—'; $('kpiFiles').textContent=fa(s.file_count||0,0);
  $('modelName').textContent=s.name || 'مدل بارگذاری‌شده'; $('tensorCount').textContent=`${fa(s.tensor_count||0,0)} Tensor`;
  $('sideModel').querySelector('strong').textContent=s.name || 'مدل بارگذاری‌شده';
  $('sideModel').querySelector('small').textContent=`${s.d_model||'—'} بُعد · ${s.layers||'—'} بلوک`;
  $('sideModel').querySelector('.status-dot').classList.add('ready');
  $('loadBanner').classList.add('hidden'); $('exportBtn').disabled=false;
  $('undoBtn').disabled=!(s.undo_count>0); $('redoBtn').disabled=!(s.redo_count>0);
  $('safeMode').checked=!!s.safe_mode;
  const evalButton=$('evaluateBtn'), evalNote=$('evalCapabilityNote');
  if(evalButton){evalButton.disabled=!s.model_executable;evalButton.title=s.model_executable?'':'برای این معماری هنوز موتور اجرای واقعی نصب نشده است.'}
  if(evalNote)evalNote.textContent=s.model_executable?'نسخه اصلی از checkpoint اولیه در حافظه ساخته می‌شود؛ فایل مبدأ تغییر نمی‌کند. مقایسه شامل تکرار، تنوع، طول و آنتروپی متن است.':'این مدل در حالت عمومی بررسی/ویرایش است. مقایسه عملکرد متنی غیرفعال است؛ پس از ویرایش، گزارش اختلاف واقعی وزن‌ها در همین صفحه نمایش داده می‌شود.';
  const activationBtn=$('activationBtn'); if(activationBtn) activationBtn.disabled=!s.activation_supported;
  const ckpt=s.files?.find(f=>['model.pt','model.pth','pytorch_model.pt','checkpoint.pt','pytorch_model.bin','model.bin','model.ckpt','model.safetensors','pytorch_model.safetensors'].includes(f.name)||/^(model|pytorch_model)-\d+.*\.(safetensors|bin)$/.test(f.name));
  $('readinessList').innerHTML = [
    [ckpt?.name||'فایل وزن','فایل اصلی وزن مدل',!!ckpt],['tokenizer / tokenizer.json','توکنایزر واژگان',s.tokenizer_ready],['config.json / پیکربندی داخلی','پیکربندی معماری',s.files?.some(f=>f.name==='config.json')||!!s.model_profile?.config?.model_type],['مستندات همراه','اطلاعات تکمیلی مدل',s.files?.some(f=>f.name==='info.json'||f.name==='README.md')],['فایل‌های اختصاصی','دارایی‌های جانبی اختیاری',s.files?.some(f=>f.name==='morph.json'||f.name==='state.json')]
  ].map(x=>`<div class="readiness-row"><span class="ready-icon ${x[2]?'':'warn'}">${x[2]?'✓':'!'}</span><div><b>${esc(x[0])}</b><small>${x[1]}</small></div><em class="${x[2]?'':'warn'}">${x[2]?'آماده':'اختیاری / بررسی'}</em></div>`).join('');
  const n=Math.max(1,Math.min(12,Number(s.layers)||4));
  let flow=`<div class="arch-block emb"><strong>WTE</strong><small>Embedding</small></div>`;
  for(let i=0;i<n;i++) flow+=`<div class="arch-block ${i%2?'mlp':'att'}"><strong>بلوک ${fa(i,0)}</strong><small>Attn · MLP</small></div>`;
  flow+=`<div class="arch-block out"><strong>Head</strong><small>خروجی</small></div>`;
  $('architectureFlow').innerHTML=flow;
  const files=(s.files||[]).slice(0,9);
  $('fileTable').innerHTML=files.length?files.map(f=>`<tr><td dir="ltr" style="text-align:right">${esc(f.name)}</td><td><span class="file-type">${esc(f.kind)}</span></td><td>${fmtBytes(f.size)}</td><td><span class="table-status">شناسایی شد</span></td></tr>`).join(''):`<tr><td colspan="4" class="empty-cell">فایل همراهی پیدا نشد.</td></tr>`;
}
function openLoad(){ $('loadModal').classList.remove('hidden'); }
$('openLoad').addEventListener('click',openLoad); $('bannerLoad').addEventListener('click',openLoad); $('closeLoad').addEventListener('click',()=>$('loadModal').classList.add('hidden'));
$('loadModal').addEventListener('click',e=>{if(e.target===$('loadModal')) $('loadModal').classList.add('hidden')});
$('chooseFolder').addEventListener('click',()=>$('folderInput').click());
$('folderInput').addEventListener('change',()=>{
  const files=[...$('folderInput').files];
  $('selectedFiles').textContent=files.length?`${fa(files.length,0)} فایل انتخاب شد · ${files.some(f=>(f.webkitRelativePath||f.name).split('/').at(-1)==='config.json')?'پیکربندی Hugging Face پیدا شد':files.some(f=>/\.(pt|pth|bin|ckpt|safetensors)$/i.test(f.name))?'فایل وزن پیدا شد':'فایل وزن هنوز شناسایی نشده'}`:'هنوز پوشه‌ای انتخاب نشده است.';
  $('uploadFolder').disabled=!files.length;
});
const drop=$('folderDrop');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('dragover')}); drop.addEventListener('dragleave',()=>drop.classList.remove('dragover'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('dragover');toast('برای حفظ ساختار پوشه، از دکمه «انتخاب پوشه مدل» استفاده کن.','error')});
$('uploadFolder').addEventListener('click',async()=>{
  const files=[...$('folderInput').files]; if(!files.length)return;
  await runBusy(async()=>{
    const fd=new FormData(); fd.append('relative_paths',JSON.stringify(files.map(f=>f.webkitRelativePath||f.name)));
    files.forEach(f=>fd.append('files',f,f.name));
    const r=await api('/api/load-folder',{method:'POST',body:fd});
    state.sid=r.session_id; localStorage.setItem('wsl_session',state.sid); state.summary=r.summary; state.proposal=null; state.selectedTensor='';
    renderSummary(r.summary); $('loadModal').classList.add('hidden'); $('folderInput').value=''; $('uploadFolder').disabled=true; $('selectedFiles').textContent='پوشه با موفقیت بارگذاری شد.';
    toast('مدل با موفقیت در حافظه بارگذاری شد.'); await loadTensors(); await loadHistory();
  },'بارگذاری پوشه مدل','فایل‌های مدل و توکنایزر بارگذاری و معماری به‌صورت خودکار شناسایی می‌شود.').catch(e=>toast(e.message,'error'));
});
$('loadLocal').addEventListener('click',async()=>{
  const path=$('localPath').value.trim(); if(!path){toast('مسیر پوشه را وارد کن.','error');return;}
  await runBusy(async()=>{
    const r=await post('/api/load-local-folder',{path}); state.sid=r.session_id;localStorage.setItem('wsl_session',state.sid);renderSummary(r.summary);$('loadModal').classList.add('hidden');state.proposal=null;await loadTensors();await loadHistory();toast('پوشه محلی مدل بارگذاری شد.');
  },'خواندن پوشه محلی').catch(e=>toast(e.message,'error'));
});
$('loadHub').addEventListener('click',async()=>{
  const model_id=$('hubModelId').value.trim();
  if(!model_id){toast('شناسه مدل Hugging Face را وارد کن.','error');return;}
  await runBusy(async()=>{
    const r=await post('/api/load-hf',{model_id,revision:$('hubRevision').value.trim(),token:$('hubToken').value.trim()});
    state.sid=r.session_id;localStorage.setItem('wsl_session',state.sid);state.proposal=null;state.selectedTensor='';renderSummary(r.summary);
    $('loadModal').classList.add('hidden');$('hubToken').value='';await loadTensors();await loadHistory();toast('مدل Hugging Face دریافت و بارگذاری شد.');
  },'دریافت مدل از Hugging Face','فایل‌های مخزن و وزن‌های مدل دریافت می‌شوند؛ زمان دانلود به اندازه مدل و سرعت اتصال بستگی دارد.').catch(e=>toast(e.message,'error'));
});

async function loadTensors(){
  if(!state.sid)return;
  try{
    const q=$('tensorSearch').value||'';const r=await api(`/api/tensors?q=${encodeURIComponent(q)}&limit=2000`);state.tensors=r.items;
    const items=$('trainableOnly').checked?r.items.filter(x=>x.trainable):r.items;
    $('tensorList').innerHTML=items.length?items.map(t=>`<button class="tensor-item ${state.selectedTensor===t.name?'active':''}" data-tensor="${esc(t.name)}"><span class="tensor-symbol">W</span><span class="tensor-text"><b>${esc(t.name)}</b><small>${esc(t.role)}</small></span><em>${esc(t.dtype)} · ${fa(t.elements,0)}</em></button>`).join(''):`<div class="empty-state">وزنی مطابق جست‌وجو پیدا نشد.</div>`;
    $('tensorCount').textContent=`${fa(r.total,0)} Tensor`;
    document.querySelectorAll('.tensor-item').forEach(b=>b.addEventListener('click',()=>selectTensor(b.dataset.tensor)));
    fillTensorSelect();
  }catch(e){toast(e.message,'error')}
}
let tensorSearchTimer;
$('tensorSearch').addEventListener('input',()=>{clearTimeout(tensorSearchTimer);tensorSearchTimer=setTimeout(loadTensors,220)});
$('clearTensorSearch').addEventListener('click',()=>$('tensorSearch').value='');$('clearTensorSearch').addEventListener('click',loadTensors);
$('trainableOnly').addEventListener('change',loadTensors);
async function selectTensor(name){
  if(!name)return;state.selectedTensor=name;
  try{
    const t=await api(`/api/tensor/${encodeURIComponent(name)}`);
    $('tensorEmpty').classList.add('hidden');$('tensorDetail').classList.remove('hidden');
    $('tensorName').textContent=t.name;$('tensorRole').textContent=t.role;$('tensorDescription').textContent=t.description;
    $('tensorMeta').innerHTML=[`ابعاد: ${t.shape.join(' × ')}`,`نوع: ${t.dtype}`,`نقش: ${t.role}`,`${t.stats.elements.toLocaleString('fa-IR')} عنصر`].map(x=>`<span>${esc(x)}</span>`).join('');
    $('statMean').textContent=fa(t.stats.mean,5);$('statStd').textContent=fa(t.stats.std,5);$('statRange').textContent=`${fa(t.stats.min,4)} / ${fa(t.stats.max,4)}`;$('statZero').textContent=pct(t.stats.zero_ratio);$('statNonfinite').textContent=fa(t.stats.nonfinite,0);$('statNorm').textContent=fa(t.stats.norm,4);
    drawHistogram($('tensorHistogram'),t.sample,'#63a7fa');$('editTensorInfo').textContent=`${t.role} · ابعاد ${t.shape.join(' × ')} · ${fa(t.stats.elements,0)} عنصر`;
    $('editTensor').value=name; $('editStart').value=0; $('editEnd').value=t.stats.elements;
    document.querySelectorAll('.tensor-item').forEach(b=>b.classList.toggle('active',b.dataset.tensor===name));
  }catch(e){toast(e.message,'error')}
}
$('editThisTensor').addEventListener('click',()=>{setPage('editor');$('editTensor').value=state.selectedTensor; $('editTensorInfo').textContent=state.selectedTensor;});
function fillTensorSelect(){
  if(!state.tensors.length)return;
  const current=$('editTensor').value||state.selectedTensor;
  $('editTensor').innerHTML=state.tensors.filter(t=>t.trainable).map(t=>`<option value="${esc(t.name)}">${esc(t.name)} — ${esc(t.role)}</option>`).join('');
  if(current && [...$('editTensor').options].some(o=>o.value===current))$('editTensor').value=current;
}
$('editTensor').addEventListener('change',()=>{state.selectedTensor=$('editTensor').value;const t=state.tensors.find(x=>x.name===state.selectedTensor);$('editTensorInfo').textContent=t?`${t.role} · ${t.shape.join(' × ')} · ${fa(t.elements,0)} عنصر`:'';});
function buildEdit(){
  const type=$('editType').value,name=$('editTensor').value;const e={tensor:name,type};
  if(type==='scale'||type==='scale_slice')e.factor=Number($('editFactor').value);
  if(type==='scale_slice'){e.start=Number($('editStart').value);e.end=Number($('editEnd').value)}
  if(type==='noise')e.std_ratio=Number($('editNoise').value);
  if(type==='clamp'){e.min_value=Number($('editMin').value);e.max_value=Number($('editMax').value)}
  return e;
}
$('editType').addEventListener('change',()=>{
  const v=$('editType').value;
  $('scaleFields').classList.toggle('hidden',!['scale','scale_slice'].includes(v));
  $('sliceFields').classList.toggle('hidden',v!=='scale_slice');$('noiseFields').classList.toggle('hidden',v!=='noise');$('clampFields').classList.toggle('hidden',v!=='clamp');
});
$('previewBtn').addEventListener('click',async()=>{
  try{
    const edit=buildEdit();const r=await post('/api/preview',{edit,safe_mode:$('safeMode').checked});state.preview=r;
    $('previewEmpty').classList.add('hidden');$('previewContent').classList.remove('hidden');
    $('diffChanged').textContent=pct(r.changed_fraction);$('diffNorm').textContent=fa(r.delta_norm,5);$('diffMean').textContent=fa(r.after.mean-r.before.mean,6);
    $('beforeStats').innerHTML=`<p>میانگین: ${fa(r.before.mean,5)}</p><p>انحراف معیار: ${fa(r.before.std,5)}</p><p>بازه: ${fa(r.before.min,4)} تا ${fa(r.before.max,4)}</p>`;
    $('afterStats').innerHTML=`<p>میانگین: ${fa(r.after.mean,5)}</p><p>انحراف معیار: ${fa(r.after.std,5)}</p><p>بازه: ${fa(r.after.min,4)} تا ${fa(r.after.max,4)}</p>`;
    drawCompareHistogram($('previewHistogram'),r.before_sample,r.after_sample);$('applyBtn').disabled=false;
  }catch(e){toast(e.message,'error');$('applyBtn').disabled=true}
});
$('applyBtn').addEventListener('click',async()=>{
  if(!state.preview)return;
  try{const r=await post('/api/apply',{edits:[buildEdit()],safe_mode:$('safeMode').checked,label:'ویرایش دستی'});renderSummary(r.summary);$('applyBtn').disabled=true;state.preview=null;toast('ویرایش با Snapshot ثبت و اعمال شد.');await loadHistory();await selectTensor(state.selectedTensor)}catch(e){toast(e.message,'error')}
});
async function runUndo(){try{const r=await post('/api/undo');renderSummary(r.summary);toast(`بازگردانی انجام شد: ${r.label}`);await loadHistory();if(state.selectedTensor)await selectTensor(state.selectedTensor)}catch(e){toast(e.message,'error')}}
async function runRedo(){try{const r=await post('/api/redo');renderSummary(r.summary);toast(`تغییر دوباره اعمال شد: ${r.label}`);await loadHistory();if(state.selectedTensor)await selectTensor(state.selectedTensor)}catch(e){toast(e.message,'error')}}
$('undoBtn').addEventListener('click',runUndo);$('redoBtn').addEventListener('click',runRedo);$('historyUndo').addEventListener('click',runUndo);$('historyRedo').addEventListener('click',runRedo);
function parseMacroText(){
  const raw=JSON.parse($('macroJson').value);
  const edits=Array.isArray(raw)?raw:raw.edits;
  if(!Array.isArray(edits)||!edits.length)throw new Error('ماکرو باید یک آرایه edits غیرخالی داشته باشد.');
  return edits;
}
$('loadMacroBtn').addEventListener('click',()=>$('macroFile').click());
$('macroFile').addEventListener('change',async()=>{
  const file=$('macroFile').files?.[0];if(!file)return;
  try{$('macroJson').value=await file.text();$('macroStatus').textContent=`فایل ${file.name} خوانده شد؛ هنوز اعتبارسنجی نشده است.`;}
  catch(e){toast(e.message,'error')}
});
$('validateMacroBtn').addEventListener('click',async()=>{
  try{const edits=parseMacroText();const r=await post('/api/validate-macro',{edits,safe_mode:$('safeMode').checked});
    $('macroStatus').textContent=`معتبر: ${fa(r.count,0)} عملیات · هیچ وزنی هنوز تغییر نکرده است.`;
    $('macroStatus').style.color='#6fe0b4';$('macroJson').value=JSON.stringify({edits:r.edits},null,2);
    toast('تمام عملیات ماکرو معتبر هستند.');
  }catch(e){$('macroStatus').textContent=e.message;$('macroStatus').style.color='#ff9aaa';toast(e.message,'error')}
});
$('applyMacroBtn').addEventListener('click',async()=>{
  try{const edits=parseMacroText();const checked=await post('/api/validate-macro',{edits,safe_mode:$('safeMode').checked});
    const r=await post('/api/apply',{edits:checked.edits,safe_mode:$('safeMode').checked,label:'اجرای ماکرو'});
    renderSummary(r.summary);$('macroStatus').textContent=`اعمال شد: ${fa(checked.count,0)} عملیات در یک Snapshot.`;$('macroStatus').style.color='#6fe0b4';
    await loadHistory();toast('ماکرو با موفقیت و در یک Snapshot اعمال شد.');
  }catch(e){$('macroStatus').textContent=e.message;$('macroStatus').style.color='#ff9aaa';toast(e.message,'error')}
});

$('resetBtn').addEventListener('click',async()=>{if(!confirm('تمام تغییرهای فعلی حذف و وزن‌ها به checkpoint اصلی بازگردانده شوند؟'))return;try{const r=await post('/api/reset');renderSummary(r.summary);await loadHistory();if(state.selectedTensor)await selectTensor(state.selectedTensor);toast('مدل به وزن‌های اصلی بازگشت.')}catch(e){toast(e.message,'error')}});
async function loadHistory(){if(!state.sid)return;try{const r=await api('/api/history');$('historyList').innerHTML=r.items.length?[...r.items].reverse().map(x=>`<div class="history-row"><span class="history-no">${fa(x.index,0)}</span><div><b>${esc(x.label)}</b><small>${x.tensors.map(esc).join('، ')}</small></div><em>${fa(x.edits.length,0)} عملیات</em></div>`).join(''):'<div class="empty-mini">هنوز تغییری ثبت نشده است.</div>';$('undoBtn').disabled=!r.undo;$('redoBtn').disabled=!r.redo;$('historyUndo').disabled=!r.undo;$('historyRedo').disabled=!r.redo;}catch{}}

$('analyzeBtn').addEventListener('click',async()=>{
  try{const r=await runBusy(()=>post('/api/analyze'),'تحلیل ساختاری مدل','وزن‌ها و ساختار معماری بررسی می‌شوند.');renderFindings(r.findings);renderSummary(r.summary);toast(`تحلیل کامل شد؛ ${fa(r.count,0)} یافته ثبت شد.`)}catch(e){toast(e.message,'error')}
});
function renderFindings(items){
  const critical=items.filter(x=>x.severity==='بحرانی').length,review=items.filter(x=>x.severity==='نیازمند بررسی').length,info=items.length-critical-review;
  $('findingTotal').textContent=fa(items.length,0);$('criticalCount').textContent=fa(critical,0);$('reviewCount').textContent=fa(review,0);$('infoCount').textContent=fa(info,0);
  const max=Math.max(1,items.length);$('criticalBar').style.width=`${critical/max*100}%`;$('reviewBar').style.width=`${review/max*100}%`;$('infoBar').style.width=`${info/max*100}%`;
  const score=Math.round(100-(critical*25+review*8+info*2));$('scoreValue').textContent=`${Math.max(0,score)}`;$('scoreRing').style.background=`conic-gradient(var(--green) ${Math.max(0,score)*3.6}deg,#233951 0)`;
  $('findingsLabel').textContent=`${fa(items.length,0)} یافته`; 
  $('findingsList').innerHTML=items.length?items.map((f,i)=>{const cl=f.severity==='بحرانی'?'critical':f.severity==='اطلاعاتی'?'info':'review';return `<article class="finding-card"><i class="finding-strip ${cl}"></i><div><h4>${esc(f.title)}</h4><p>${esc(f.detail)}</p><small>پیشنهاد بررسی: ${esc(f.recommendation)}</small></div><button class="btn btn-outline btn-sm finding-action" data-finding-tensor="${esc(f.tensor||'')}">${f.tensor?'مشاهده وزن':'جزئیات'}</button></article>`}).join(''):'<div class="empty-mini">یافته‌ای ثبت نشد.</div>';
  document.querySelectorAll('[data-finding-tensor]').forEach(b=>b.addEventListener('click',()=>{if(b.dataset.findingTensor){setPage('tensors');$('tensorSearch').value=b.dataset.findingTensor;loadTensors().then(()=>selectTensor(b.dataset.findingTensor))}}));
}
$('activationBtn').addEventListener('click',async()=>{try{const r=await runBusy(()=>post('/api/activation',{prompt:$('activationPrompt').value}),'بررسی فعال‌سازی','یک گذر واقعی از شبکه در حال اجراست.');$('activationResults').classList.remove('hidden');$('activationTable').innerHTML=r.layers.map(x=>`<tr><td>بلوک ${fa(x.layer,0)}</td><td>${fa(x.mean,4)}</td><td>${fa(x.std,4)}</td><td>${fa(x.rms,4)}</td><td>${fa(x.max_abs,4)}</td><td>${fa(x.proxy_entropy,4)}</td></tr>`).join('');toast('بررسی فعال‌سازی پایان یافت.')}catch(e){toast(e.message,'error')}});

$('symptomGrid').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{$('symptomGrid').querySelectorAll('button').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');state.selectedSymptom=b.dataset.symptom}));
$('guidedBtn').addEventListener('click',async()=>{try{const r=await post('/api/guided',{symptom:state.selectedSymptom});$('guidedCandidates').classList.remove('hidden');$('guidedCandidates').innerHTML=r.candidates.length?r.candidates.map(c=>`<div class="candidate-row"><div><code>${esc(c.name)}</code><span>${esc(c.role)} · ${esc(c.shape.join(' × '))}</span></div><button class="btn btn-outline btn-sm" data-candidate="${esc(c.name)}">انتخاب</button></div>`).join(''):'<div class="empty-mini">نامزد مرتبطی در این گروه پیدا نشد.</div>';document.querySelectorAll('[data-candidate]').forEach(b=>b.addEventListener('click',()=>{state.selectedTensor=b.dataset.candidate;$('editTensor').value=b.dataset.candidate;$('editTensorInfo').textContent=b.dataset.candidate;setPage('editor');selectTensor(b.dataset.candidate)}))}catch(e){toast(e.message,'error')}});
$('symptomGrid').querySelectorAll('button').forEach(()=>{});
document.querySelectorAll('.mode-option').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.mode-option').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.mode=b.dataset.mode;$('aiConfig').classList.toggle('hidden',state.mode!=='ai')}));
$('safeMode').addEventListener('change',async()=>{if(!state.sid)return;try{await post('/api/settings',{safe_mode:$('safeMode').checked,base_url:$('apiBase').value,api_key:$('apiKey').value,model:$('apiModel').value});toast($('safeMode').checked?'حالت ایمن فعال شد.':'حالت ایمن غیرفعال شد؛ ریسک تغییرها بیشتر می‌شود.','success')}catch(e){toast(e.message,'error')}});
$('saveApi').addEventListener('click',async()=>{try{await post('/api/settings',{base_url:$('apiBase').value,api_key:$('apiKey').value,model:$('apiModel').value,safe_mode:$('safeMode').checked});toast('تنظیمات تحلیل‌گر ذخیره شد.')}catch(e){toast(e.message,'error')}});
$('proposalBtn').addEventListener('click',async()=>{
  try{
    const prompts=$('planPrompts').value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);
    const goal=$('planGoal').value.trim()||'بررسی جامع کل مدل و شناسایی فرصت‌های بهبود بدون فرض وجود مشکل از پیش تعیین‌شده';
    const r=await runBusy(()=>post('/api/proposal',{symptom:goal,mode:state.mode,full_plan:true,prompts}),'شناخت و تحلیل جامع مدل','همه تنسورهای قابل‌آموزش، نقش معماری، آمار وزن، یافته‌ها، فعال‌سازی و نمونه خروجی‌ها برای تحلیل‌گر آماده می‌شوند.');
    state.proposal=r.proposal;renderProposal(r.proposal);toast('برنامه جامع آماده شد؛ همه تغییرها را پیش از اعمال مرور کن.');
  }catch(e){toast(e.message,'error')}
});
function renderProposal(p){
  const risk={low:'کم',medium:'متوسط',high:'زیاد'};
  const ident=p.model_identity||state.summary?.model_profile||{};
  const strengths=(p.strengths||[]).map(x=>`<div class="plan-insight"><b>${esc(x.title||'نقطه قوت')}</b><p>${esc(x.meaning||x.evidence||'')}</p><small>${esc(x.evidence||'')}</small></div>`).join('');
  const weaknesses=(p.weaknesses||[]).map(x=>`<div class="plan-insight"><b>${esc(x.title||'مورد نیازمند بررسی')} <span class="certainty">${esc(x.certainty==='observed'?'مشاهده‌شده':x.certainty==='hypothesis'?'فرضیه':'نامشخص')}</span></b><p>${esc(x.meaning||'')}</p><small>شاهد: ${esc(x.evidence||'—')}</small><small>بررسی بعدی: ${esc(x.what_to_check||'—')}</small></div>`).join('');
  const actions=(p.optimization_plan||[]).map(x=>`<div class="plan-action"><span class="priority">${fa(x.priority||0,0)}</span><div><b>${esc(x.title||'اقدام پیشنهادی')}</b><p>${esc(x.goal||'')}</p><small>${esc(x.why||'')} · روش: ${esc(x.method||'')} · سنجش: ${esc(x.validation||'')}</small></div><em class="risk-${esc(x.risk||'medium')}">ریسک ${risk[x.risk]||'متوسط'}</em></div>`).join('');
  const edits=p.suggested_edits||[];
  const editRows=edits.map((e,i)=>`<div class="edit-proposal plan-edit"><div><code>${esc(e.tensor)}</code><p>${esc(e.reason||'ویرایش آزمایشی')} · ریسک ${risk[e.risk]||'متوسط'} · ارتباط با هدف ${pct(e.target_relevance||0)}</p><small>اثر مورد انتظار: ${esc(e.expected_effect||'در آزمون مشخص می‌شود.')}</small><small>اثر جانبی: ${esc(e.possible_side_effect||'ممکن است کیفیت بخش دیگری افت کند.')}</small><small>روش سنجش: ${esc(e.validation||'مقایسه با پرامپت‌های ثابت')}</small></div><span class="edit-index">${fa(i+1,0)}</span></div>`).join('');
  const nonWeight=(p.non_weight_actions||[]).map(x=>`<div class="plan-insight"><b>${esc(x.title||'اقدام تکمیلی')}</b><p>${esc(x.why||'')}</p><small>${(x.steps||[]).map(esc).join(' ← ')}</small></div>`).join('');
  const notRecommended=(p.not_recommended||[]).map(x=>`<li><b>${esc(x.action||'')}</b> — ${esc(x.reason||'')}</li>`).join('');
  $('proposalPanel').innerHTML=`<div class="full-plan-card"><div class="full-plan-head"><div><span class="section-kicker">WHOLE MODEL REPORT / ${state.mode==='ai'?'LLM ANALYSIS':'LOCAL'}</span><h2>گزارش جامع شناخت و بهبود مدل</h2><p>${esc(p.executive_summary||p.hypothesis||'تحلیل کامل مدل آماده شد.')}</p></div><div class="plan-score"><b>${pct(p.confidence||0)}</b><small>اطمینان کلی گزارش</small></div></div>
  <div class="plan-model-id"><span>مدل شناسایی‌شده</span><b>${esc(ident.name||state.summary?.name||'مدل')}</b><strong>${esc(ident.task||ident.purpose||'وظیفه از روی معماری استنباط شده است.')}</strong><small>${esc(ident.architecture||ident.family||'')}</small></div>
  <div class="plan-section"><h3>نقاط قوت شناسایی‌شده</h3><div class="plan-insights">${strengths||'<div class="empty-mini">تحلیل‌گر نقطه قوت مشخصی گزارش نکرده است.</div>'}</div></div>
  <div class="plan-section"><h3>ضعف‌ها و موارد نیازمند بررسی</h3><div class="plan-insights">${weaknesses||'<div class="empty-mini">ضعف مشخصی با شواهد کافی شناسایی نشد.</div>'}</div></div>
  <div class="plan-section"><h3>برنامه اولویت‌بندی‌شده بهبود</h3><div class="plan-actions">${actions||'<div class="empty-mini">اقدام مشخصی پیشنهاد نشده است.</div>'}</div></div>
  <div class="plan-section"><div class="plan-title-row"><div><h3>تغییرهای وزن پیشنهادی</h3><p>${fa(edits.length,0)} تغییر معتبر و قابل اعمال در یک تراکنش</p></div><span class="chip chip-green">${p.reversible?'قابل بازگشت':'نیازمند احتیاط'}</span></div>${editRows||'<div class="empty-mini">با شواهد فعلی، تغییر وزن قابل دفاعی پیشنهاد نشده است. این نتیجه از ساختن ویرایش حدسی بهتر است.</div>'}
  ${edits.length?'<button class="btn btn-primary btn-block" id="applyAllPlan">اعمال همه تغییرهای پیشنهادی و اجرای آزمون قبل/بعد</button>':''}</div>
  <div class="plan-section"><h3>برنامه آزمون پیشنهادی تحلیل‌گر</h3><p>این پرسش‌ها از روی هدف مدل و ضعف‌های احتمالی پیشنهاد شده‌اند و پس از پرسش‌های اولیه به آزمون قبل/بعد افزوده می‌شوند.</p><div class="benchmark-plan"><div><b>پرسش‌های آزمون</b><ol>${((p.benchmark_plan||{}).prompts||[]).map(x=>`<li>${esc(x)}</li>`).join('')||'<li>پرسش اختصاصی پیشنهاد نشده است.</li>'}</ol></div><div><b>معیارهای پیشنهادی</b><ul>${((p.benchmark_plan||{}).metrics||[]).map(x=>`<li>${esc(x)}</li>`).join('')||'<li>تکرار، تنوع، طول و آنتروپی متن</li>'}</ul><small>${esc((p.benchmark_plan||{}).missing_evidence||'')}</small></div></div><button class="btn btn-outline btn-sm" id="useBenchmarkPrompts">استفاده از پرسش‌های پیشنهادی در کادر آزمون</button></div>
  <div class="plan-section"><h3>بهبودهای بدون دستکاری وزن</h3><div class="plan-insights">${nonWeight||'<div class="empty-mini">اقدام غیر وزنی مشخصی گزارش نشده است.</div>'}</div></div>
  ${notRecommended?`<div class="plan-section"><h3>کارهایی که فعلاً توصیه نمی‌شوند</h3><ul class="not-recommended">${notRecommended}</ul></div>`:''}
  <div class="plan-limitations"><b>محدودیت‌ها و ریسک کلی · ${risk[p.risk]||'نامشخص'}</b><p>${esc(p.limitations||'این گزارش فرضیه‌محور است و بهبود عملکرد را تضمین نمی‌کند.')}</p>${(p.rejected_edits||[]).length?`<small>${fa(p.rejected_edits.length,0)} پیشنهاد به دلیل ناسازگاری با Tensorها یا محدودیت حالت ایمن رد شد.</small>`:''}</div></div>`;
  $('useBenchmarkPrompts')?.addEventListener('click',()=>{const prompts=(p.test_prompts||[]);if(prompts.length)$('planPrompts').value=prompts.join('\n');toast('پرسش‌های پیشنهادی برای آزمون آماده شدند.');});
  $('applyAllPlan')?.addEventListener('click',async()=>{
    if(!confirm(`تمام ${edits.length} تغییر پیشنهادی در یک Snapshot اعمال شوند؟ فایل اصلی تغییر نمی‌کند و امکان بازگشت وجود دارد.`))return;
    try{
      await runBusy(async()=>{
        const applied=await post('/api/apply',{edits,safe_mode:$('safeMode').checked,label:'اعمال برنامه جامع LLM'});
        renderSummary(applied.summary);await loadHistory();
        const diff=await api('/api/weight-diff');
        if(state.summary?.model_executable){
          const prompts=(p.test_prompts||$('planPrompts').value.split(/\r?\n/)).map(x=>String(x).trim()).filter(Boolean).slice(0,12);
          const ev=await post('/api/evaluate',{name:'ارزیابی خودکار پس از برنامه جامع',prompts,max_new_tokens:64,temperature:0});
          state.lastEval=ev;renderEvaluation(ev);renderWeightDiff(diff);await loadExperiments();setPage('evaluation');
        }else{
          renderWeightDiff(diff);$('evalEmpty').classList.add('hidden');$('evalContent').classList.add('hidden');$('weightDiffPanel').classList.remove('hidden');setPage('evaluation');
        }
      },'اعمال برنامه و ساخت گزارش قبل/بعد','ابتدا همه تغییرها در یک Snapshot اعمال می‌شوند؛ سپس تفاوت دقیق وزن‌ها ثبت می‌شود. اگر معماری موتور اجرایی داشته باشد، آزمون رفتاری نیز اجرا می‌شود.');
      toast(state.summary?.model_executable?'برنامه اعمال شد؛ گزارش وزن و آزمون رفتاری قبل/بعد آماده است.':'تغییرها اعمال شدند؛ گزارش دقیق وزن‌ها آماده است. سنجش عملکرد به adapter معماری نیاز دارد.');
    }catch(e){toast(`اعمال یا گزارش تغییرها کامل نشد: ${e.message}`,'error');}
  });
}


$('evaluateBtn').addEventListener('click',async()=>{
  try{const prompts=$('evalPrompts').value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);const r=await runBusy(()=>post('/api/evaluate',{name:$('experimentName').value,prompts,max_new_tokens:Number($('evalTokens').value),temperature:Number($('evalTemp').value)}),'اجرای آزمون A/B','نسخه اصلی و نسخه فعلی با پرامپت‌های یکسان اجرا می‌شوند؛ برای مدل‌های بزرگ ممکن است زمان‌بر باشد.');state.lastEval=r;renderEvaluation(r);renderWeightDiff(await api('/api/weight-diff'));await loadExperiments();toast('مقایسه A/B پایان یافت.')}catch(e){toast(e.message,'error')}
});
function renderEvaluation(r){
  $('evalEmpty').classList.add('hidden');$('evalContent').classList.remove('hidden');const s=r.summary;
  const metrics=[['تکرار سه‌تایی',s.mean_repetition_before,s.mean_repetition_after,false],['تنوع سه‌تایی',s.mean_diversity_before,s.mean_diversity_after,true],['میانگین طول پاسخ',s.mean_words_before,s.mean_words_after,true],['آنتروپی خروجی',s.mean_entropy_before,s.mean_entropy_after,true]];
  $('evalMetrics').innerHTML=metrics.map(([n,b,a,up])=>{const d=a-b;return `<div class="eval-metric"><small>${n}</small><b>${fa(a,4)}</b><em class="${(up?d>=0:d<=0)?'up':'down'}">${d>=0?'+':''}${fa(d,4)}</em><small>قبل: ${fa(b,4)}</small></div>`}).join('');
  const repDelta=s.mean_repetition_after-s.mean_repetition_before;const divDelta=s.mean_diversity_after-s.mean_diversity_before;const signal=repDelta<0?'نسبت تکرار کاهش یافته است.':repDelta>0?'نسبت تکرار بیشتر شده است.':'نسبت تکرار تقریباً بدون تغییر است.';const trade=divDelta<-.03?'هم‌زمان تنوع سه‌تایی افت کرده و باید خروجی‌ها با دقت مرور شوند.':divDelta>.03?'تنوع سه‌تایی افزایش یافته؛ این به‌تنهایی صحت یا انسجام را اثبات نمی‌کند.':'تنوع سه‌تایی تغییر بزرگی نکرده است.';const verdict=$('evalVerdict');if(verdict)verdict.innerHTML=`<b>برداشت خودکار از شاخص‌ها</b><p>${signal} ${trade}</p><small>این آزمایش معیارهای سطحی متن را می‌سنجد؛ برای سنجش دانش و صحت پاسخ باید سؤال‌های دارای جواب مرجع یا داوری انسانی اضافه شوند.</small>`;
  drawBars($('evalChart'),['تکرار','تنوع','طول','آنتروپی'],[s.mean_repetition_before,s.mean_diversity_before,s.mean_words_before,s.mean_entropy_before],[s.mean_repetition_after,s.mean_diversity_after,s.mean_words_after,s.mean_entropy_after]);
  $('evalRows').innerHTML=r.rows.map(row=>`<div class="eval-row"><b>پرامپت ${fa(row.index,0)}: ${esc(row.prompt)}</b><div class="answer-columns"><div><small>مدل اصلی</small><span>${esc(row.before.text||'(بدون خروجی)')}</span></div><div><small>نسخه ویرایش‌شده</small><span>${esc(row.after.text||'(بدون خروجی)')}</span></div></div><p>تغییر تکرار: ${fa(row.delta.repetition,4)} · تغییر تنوع: ${fa(row.delta.diversity,4)} · تغییر طول: ${fa(row.delta.words,0)} واژه</p></div>`).join('')+`<p class="note">${esc(s.note||'معیارهای متنی به تنهایی صحت معنایی را نشان نمی‌دهند.')}</p>`;
}
function renderWeightDiff(r){
  const panel=$('weightDiffPanel');if(!panel)return;
  const rows=r.items||[];
  panel.innerHTML=`<div class="panel-head"><div><h3>گزارش تغییر واقعی وزن‌ها</h3><p>مقایسه نسخه اصلی در حافظه با مدل پس از اعمال برنامه</p></div><span class="chip">${fa(r.changed_tensors||0,0)} Tensor تغییرکرده</span></div>
  <div class="diff-summary"><div><small>تعداد Tensorهای تغییرکرده</small><b>${fa(r.changed_tensors||0,0)}</b></div><div><small>تعداد مقادیر تغییرکرده</small><b>${fa(r.changed_elements||0,0)}</b></div><div><small>بیشترین تغییر مطلق</small><b>${fa(Math.max(0,...rows.map(x=>x.max_abs_delta||0)),6)}</b></div></div>
  <p class="note">${esc(r.performance_note||'')} آمار میانگین، انحراف معیار و RMS از نمونه‌گیری وزن‌ها به‌دست می‌آید؛ شمارش عناصر تغییرکرده و بیشترین اختلاف به‌صورت کامل محاسبه می‌شوند.</p>
  <div class="diff-groups"><div><b>تغییر بر اساس نقش</b>${(r.role_summary||[]).map(x=>`<p>${esc(x.name)}: ${fa(x.changed_elements,0)} مقدار</p>`).join('')||'<small>بدون تغییر</small>'}</div><div><b>تغییر بر اساس بلوک</b>${(r.layer_summary||[]).map(x=>`<p>${esc(x.name)}: ${fa(x.changed_elements,0)} مقدار</p>`).join('')||'<small>بدون تغییر</small>'}</div></div>
  ${rows.length?`<div class="diff-table-wrap"><table class="diff-table"><thead><tr><th>Tensor</th><th>نقش</th><th>تعداد تغییر</th><th>درصد تغییر</th><th>RMS قبل ← بعد</th><th>بیشینه تغییر</th></tr></thead><tbody>${rows.map(x=>`<tr><td><code>${esc(x.name)}</code><small>${esc((x.shape||[]).join(' × '))}</small></td><td>${esc(x.role||'—')}</td><td>${x.changed_elements===null?'تغییر شکل':fa(x.changed_elements,0)}</td><td>${x.changed_fraction===null?'—':pct(x.changed_fraction)}</td><td>${x.before?`${fa(x.before.rms||0,4)} ← ${fa(x.after.rms||0,4)}`:'—'}</td><td>${fa(x.max_abs_delta||0,6)}</td></tr>`).join('')}</tbody></table></div>`:'<div class="empty-mini">هیچ اختلافی میان نسخه اصلی و نسخه فعلی پیدا نشد.</div>'}`;
  panel.classList.remove('hidden');
}

async function loadExperiments(){try{const r=await api('/api/experiments');$('experimentCount').textContent=`${fa(r.items.length,0)} آزمایش`;$('experimentList').innerHTML=r.items.length?[...r.items].reverse().map(x=>`<div class="experiment-item"><span class="history-no">${fa(x.id,0)}</span><div><b>${esc(x.name)}</b><small>${fa(x.summary.prompts,0)} پرامپت · تکرار قبل ${pct(x.summary.mean_repetition_before)} · بعد ${pct(x.summary.mean_repetition_after)}</small></div><span class="chart-label">A/B</span></div>`).join(''):'<div class="empty-mini">نتیجه‌ای ثبت نشده است.</div>'}catch{}}

$('exportBtn').addEventListener('click',async()=>{
  try{const res=await api('/api/export',{method:'POST'});const blob=await res.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='WeightSpace_Harraf_Export.zip';a.click();URL.revokeObjectURL(url);toast('خروجی کامل مدل آماده شد.')}catch(e){toast(e.message,'error')}
});

function drawHistogram(canvas,values,color='#63a7fa'){
  if(!canvas)return;const ctx=canvas.getContext('2d'),w=canvas.clientWidth||500,h=canvas.height;canvas.width=w*devicePixelRatio;canvas.height=h*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,w,h);
  if(!values?.length){ctx.fillStyle='#7187a4';ctx.fillText('نمونه‌ای برای نمایش وجود ندارد',w/2-70,h/2);return}
  const min=Math.min(...values),max=Math.max(...values),bins=36,counts=Array(bins).fill(0),range=max-min||1;
  values.forEach(v=>{const i=Math.min(bins-1,Math.max(0,Math.floor((v-min)/range*bins)));counts[i]++});const peak=Math.max(1,...counts),pad=22,bw=(w-pad*2)/bins;
  ctx.strokeStyle='rgba(145,170,205,.12)';ctx.lineWidth=1;for(let i=0;i<4;i++){const y=10+i*(h-35)/3;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke()}
  counts.forEach((c,i)=>{const bh=(c/peak)*(h-42);ctx.fillStyle=color;ctx.globalAlpha=.38+.45*(c/peak);ctx.beginPath();roundRect(ctx,pad+i*bw,h-25-bh,Math.max(2,bw-2),bh,3);ctx.fill()});ctx.globalAlpha=1;
  ctx.fillStyle='#7086a4';ctx.font='10px Vazirmatn, Tahoma';ctx.fillText(fa(min,3),pad,h-7);ctx.fillText(fa(max,3),w-pad-42,h-7);
}
function drawCompareHistogram(canvas,before,after){
  if(!canvas)return;const ctx=canvas.getContext('2d'),w=canvas.clientWidth||500,h=canvas.height;canvas.width=w*devicePixelRatio;canvas.height=h*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,w,h);
  const vals=[...(before||[]),...(after||[])];if(!vals.length)return;const min=Math.min(...vals),max=Math.max(...vals),range=max-min||1,bins=35,pad=22,bw=(w-pad*2)/bins;const countsFor=a=>{const c=Array(bins).fill(0);a.forEach(v=>c[Math.min(bins-1,Math.max(0,Math.floor((v-min)/range*bins)))]++);return c};const a=countsFor(before||[]),b=countsFor(after||[]),peak=Math.max(1,...a,...b);
  for(let i=0;i<bins;i++){for(const [c,col,dx] of [[a[i],'#5b9df5',0],[b[i],'#49d4b2',bw*.48]]){const bh=c/peak*(h-38);ctx.fillStyle=col;ctx.globalAlpha=.75;ctx.fillRect(pad+i*bw+dx,h-22-bh,bw*.45,bh)}}ctx.globalAlpha=1;ctx.fillStyle='#7c92af';ctx.font='10px Vazirmatn,Tahoma';ctx.fillText('آبی: قبل',pad,12);ctx.fillText('سبز: بعد',pad+65,12);
}
function drawBars(canvas,labels,before,after){
  if(!canvas)return;const ctx=canvas.getContext('2d'),w=canvas.clientWidth||500,h=canvas.height;canvas.width=w*devicePixelRatio;canvas.height=h*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,w,h);const max=Math.max(1,...before,...after),left=20,base=h-28,slot=(w-left-15)/labels.length,bw=Math.min(24,slot*.28);
  for(let i=0;i<labels.length;i++){const x=left+i*slot+slot/2;const bh=before[i]/max*(h-60),ah=after[i]/max*(h-60);ctx.fillStyle='#578fe0';roundRect(ctx,x-bw-2,base-bh,bw,bh,4);ctx.fill();ctx.fillStyle='#46cfa8';roundRect(ctx,x+2,base-ah,bw,ah,4);ctx.fill();ctx.fillStyle='#91a6c1';ctx.font='9px Vazirmatn,Tahoma';ctx.textAlign='center';ctx.fillText(labels[i],x,base+15)}ctx.textAlign='right';ctx.fillStyle='#7d93b0';ctx.fillText('آبی: قبل',w-75,14);ctx.fillStyle='#55d6ae';ctx.fillText('سبز: بعد',w-15,14);
}
function roundRect(ctx,x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath()}

// Recover a loaded in-memory session after a browser refresh.
(async()=>{
  if(!state.sid)return;
  try{const s=await api('/api/summary');renderSummary(s);await loadTensors();await loadHistory();await loadExperiments();toast('نشست قبلی بازیابی شد.')}catch{state.sid='';localStorage.removeItem('wsl_session')}
})();
