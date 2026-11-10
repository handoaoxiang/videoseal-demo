const $ = id => document.getElementById(id);
const defaultMethods = ['dct','spectral','denoise','vae','geometry','guided','temporal'];
let methods = [], selected = [...defaultMethods], sample = false, jobs = [], activeId = localStorage.getItem('spectral-job'), renderedResult = null, uploading = false;
let pendingFiles = [], batchItems = [], maxBatch = 5, accelerationReady = false, nativeAllFrames = false;
try{batchItems=JSON.parse(localStorage.getItem('spectral-batch')||'[]');if(!Array.isArray(batchItems))batchItems=[];}catch{batchItems=[];}
const newId = () => crypto.randomUUID().replaceAll('-','');
const icons = {dct:'▦',carrier:'⌁',spectral:'≋',denoise:'◈',vae:'◇',geometry:'⌖',guided:'◎',temporal:'↝'};
const states = {queued:'排队中',running:'运行中',completed:'处理完成',failed:'失败',cancelled:'已取消',cancelling:'取消中',interrupted:'已中断'};
const running = s => ['queued','running','cancelling'].includes(s);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const friendlyError = value => /moov atom not found|Invalid data found when processing input/i.test(String(value))?'无法读取这个视频，请确认文件完整，并能在本机播放器正常播放。':String(value);
function toast(message){ $('toast').textContent=message; $('toast').hidden=false; clearTimeout(toast.timer); toast.timer=setTimeout(()=>$('toast').hidden=true,3500); }
async function api(url, options={}){ const response=await fetch(url,options); if(!response.ok){let message;try{const body=await response.json();message=body.detail;}catch{message='服务暂时不可用';}throw new Error(friendlyError(typeof message==='string'?message:JSON.stringify(message)));}return response.json(); }
async function loadCapabilities(){
  try{const data=await api('/api/capabilities'); methods=data.methods;nativeAllFrames=(data.max_processing_frames||0)>=9000;accelerationReady=!!data.adaptive_vae;$('vae-mode').querySelector('option[value="fast"]').disabled=!data.fast_vae;$('vae-mode').disabled=!accelerationReady;$('denoise-device').disabled=!accelerationReady;renderVaeNote();maxBatch=data.max_pending_jobs||5;renderMethods();const nativeSizeReady=(data.max_processing_side||1280)>=1920;$('max-side').querySelector('option[value="1920"]').disabled=!nativeSizeReady;$('size-note').textContent=nativeSizeReady?'原片为 1080 × 1920 时，选择 1920 px、30 FPS 可保留原分辨率和帧率。尺寸越大处理越慢；编码后画质和码率仍可能变化。':'1920 px 选项需要新版后端。请先停止旧网页服务，再双击 run_web.cmd 并刷新页面；刷新环境状态后即可选择。';$('upload-note').textContent=data.max_pending_jobs?`每批最多 ${maxBatch} 个视频，使用同一组参数，逐个上传、串行处理。上传结束前请保持页面打开。`:'当前后端尚未更新：请重启 run_web.cmd 以启用 50 个任务队列和上传重试去重。目前每批最多 5 个视频，上传结束前请保持页面打开。';$('runtime').textContent=data.ffmpeg?'本地引擎已连接':'FFmpeg 尚未就绪';$('use-sample').disabled=uploading||!data.sample_ready;$('device-tag').textContent=data.acceleration?.cuda?'CUDA':'CPU';$('model-state').textContent=`${data.device} · ${data.acceleration?.nvenc?'NVENC 编码可用':'CPU 编码'} · VAE ${methods.find(m=>m.id==='vae').ready?'已就绪':'待安装模型'} · ${data.acceleration?.denoise_opencl?.available?'GPU 去噪可用':'CPU 去噪'}`;}
  catch(error){$('runtime').textContent='本地服务未连接';toast(error.message);}
}
function renderMethods(){
  $('method-grid').innerHTML=methods.map(m=>`<button type="button" class="method-card ${selected.includes(m.id)?'selected':''} ${!m.ready?'unavailable':''}" data-method="${m.id}" aria-pressed="${selected.includes(m.id)}" title="${escapeHtml(m.condition)}"><div class="method-top"><span class="method-icon">${icons[m.id]}</span><span class="method-title">${escapeHtml(m.name)}</span><span class="method-check">✓</span></div><p>${escapeHtml(m.subtitle)}</p><div class="method-foot"><span>${m.ready?escapeHtml(m.condition):'未就绪 · 点击查看配置'}</span><span class="method-group">${m.group}</span></div></button>`).join('');
  $('pipeline').innerHTML=selected.length?selected.map((id,index)=>`<span class="pipeline-chip">${index+1}. ${escapeHtml(methods.find(m=>m.id===id)?.name||id)}<button data-remove="${id}" aria-label="移除${escapeHtml(methods.find(m=>m.id===id)?.name||id)}">×</button></span>`).join(''):'请选择至少一种方法';
  $('reference-fields').hidden=!selected.includes('carrier');
}
$('method-grid').addEventListener('click',event=>{const card=event.target.closest('[data-method]');if(!card)return;const method=methods.find(m=>m.id===card.dataset.method);if(!method.ready){toast('VAE 需要本地模型，请运行 setup_web.ps1 -WithVae 后刷新环境状态。');return;}selected=selected.includes(method.id)?selected.filter(x=>x!==method.id):[...selected,method.id];renderMethods();});
$('pipeline').addEventListener('click',event=>{const button=event.target.closest('[data-remove]');if(button){selected=selected.filter(x=>x!==button.dataset.remove);renderMethods();}});
function renderFiles(){
  $('selected-files').hidden=!pendingFiles.length;
  $('selected-files').innerHTML=pendingFiles.map((entry,index)=>`<div class="selected-file"><span>${escapeHtml(entry.file.name)}<small>${(entry.file.size/1024/1024).toFixed(1)} MB</small></span><button class="text-button" data-remove-file="${index}" ${uploading?'disabled':''} aria-label="移除 ${escapeHtml(entry.file.name)}">移除</button></div>`).join('');
  if(!sample){$('file-title').textContent=pendingFiles.length?`已选择 ${pendingFiles.length} 个视频`:'拖入多个视频，或点击批量选择';$('file-meta').textContent=pendingFiles.length?`共 ${(pendingFiles.reduce((sum,x)=>sum+x.file.size,0)/1024/1024).toFixed(1)} MB · 点击可继续添加`:'支持多选、分次添加 · 单个文件最大 512 MB';$('drop-zone').classList.toggle('selected',!!pendingFiles.length);}
}
function addFiles(files){
  if(uploading)return;
  const entries=[...files].filter(file=>!pendingFiles.some(x=>x.file.name===file.name&&x.file.size===file.size&&x.file.lastModified===file.lastModified));
  if(pendingFiles.length+entries.length>maxBatch){toast(`每次最多选择 ${maxBatch} 个视频，请分批添加。`);return;}
  sample=false;pendingFiles.push(...entries.map(file=>({file,requestId:newId()})));$('use-sample').textContent='使用 VideoSeal 验证样本 →';renderFiles();
}
$('video').addEventListener('change',()=>{addFiles($('video').files);$('video').value='';});
$('selected-files').onclick=event=>{const button=event.target.closest('[data-remove-file]');if(button&&!uploading){pendingFiles.splice(Number(button.dataset.removeFile),1);renderFiles();}};
$('drop-zone').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('video').click();}});
for(const type of ['dragenter','dragover'])$('drop-zone').addEventListener(type,event=>{event.preventDefault();$('drop-zone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('drop-zone').addEventListener(type,event=>{event.preventDefault();$('drop-zone').classList.remove('dragging');});
$('drop-zone').addEventListener('drop',event=>addFiles(event.dataTransfer.files));
$('use-sample').onclick=()=>{if(uploading)return;sample=true;pendingFiles=[];renderFiles();$('video').value='';$('file-title').textContent='VideoSeal 已知水印验证样本';$('file-meta').textContent='2 秒 · 256 × 256 · 256 位参考消息自动载入';$('drop-zone').classList.add('selected');$('use-sample').textContent='✓ 已选择验证样本';$('seconds').value='2';$('max-side').value='256';$('fps').value='4';toast('已选择样本，点击「开始实验」运行。');};
$('strength').oninput=()=>$('strength-value').textContent=Number($('strength').value).toFixed(1);
function renderSampling(){const random=$('frame-mode').value==='random';$('random-frame-settings').hidden=!random;$('seconds').querySelector('option[value="300"]').textContent=random?'前 300 秒（随机模式最多 9000 帧）':'前 300 秒（最多 9000 帧）';$('sampling-note').textContent=random?'在所选时长内随机选 N 帧执行所选方法，其余帧跳过算法；输出仍覆盖所选时长，尺寸、帧率和编码设置仍生效。相同种子和相同帧数可复测。抽帧切换可能闪烁，不保证清除水印。随机模式最多输出 9000 帧。':'对所选时长内的每一帧执行所选方法，单次最多 9000 帧。';}
$('frame-mode').onchange=renderSampling;
function renderExport(){$('export-note').textContent=$('export-mode').value==='full'?'导出原视频完整时长：前段执行所选方法，后段不执行算法。后段仍按相同尺寸、帧率重新编码，完整音频保留。对照预览仅包含前段。':'只导出上面选择的时长。例如选择前 2 秒，下载的视频约 2 秒。';}
$('export-mode').onchange=renderExport;
function renderVaeNote(){$('acceleration-note').textContent=!accelerationReady?'当前服务需要重启后才能启用 VAE 自适应分块和 GPU 去噪；请停止旧网页服务，再运行 run_web.cmd。':$('vae-mode').value==='fast'?'快速模式使用轻量模型重建，保持输出尺寸和 FP32 精度。重建效果与原模型不同，细节和颜色可能变化；处理完成仍不代表未知水印已清除。':'自动模式保持所选分辨率和 FP32 精度。不同分块方式及 GPU 去噪与旧版可能有细微像素差异，可选择兼容方式做对照。';}
$('vae-mode').onchange=renderVaeNote;
$('reset-settings').onclick=()=>{selected=[...defaultMethods];renderMethods();for(const[id,value]of Object.entries({strength:3,seconds:10,'max-side':640,fps:8,crf:18,steps:6,'frequency-mode':'dct',band:'mid_high','compute-device':'auto',encoder:'auto','vae-mode':'auto','denoise-device':'auto','frame-mode':'all','random-frames':10,'random-seed':42,'export-mode':'clip'}))$(id).value=value;$('strength').oninput();renderSampling();renderExport();renderVaeNote();};
$('refresh-models').onclick=()=>loadCapabilities();
$('run').onclick=async()=>{
  if(uploading)return;$('form-error').hidden=true;
  try{
    if(!selected.length)throw new Error('请选择至少一种处理方法。');
    if(!sample&&!pendingFiles.length)throw new Error('请先选择视频，或使用验证样本。');
    const unavailable=selected.filter(id=>!methods.find(m=>m.id===id)?.ready);if(unavailable.length)throw new Error('所选功能尚未就绪，请取消相应功能或准备本地模型后重试。');
    if(selected.includes('carrier')&&(!$('reference-clean').files.length||!$('reference-marked').files.length))throw new Error('载波抵消需要两份对齐的参考视频。');
    const settings={methods:[...selected],strength:Number($('strength').value),seconds:Number($('seconds').value),max_side:Number($('max-side').value),fps:Number($('fps').value),crf:Number($('crf').value),steps:Number($('steps').value),frequency_mode:$('frequency-mode').value,band:$('band').value,message_bits:$('message-bits').value.trim(),compute_device:$('compute-device').value,encoder:$('encoder').value};
    if(accelerationReady){settings.vae_mode=$('vae-mode').value;settings.denoise_device=$('denoise-device').value;}
    settings.frame_mode=$('frame-mode').value;
    settings.export_mode=$('export-mode').value;
    if(settings.frame_mode==='random'){
      settings.random_frames=Number($('random-frames').value);settings.random_seed=Number($('random-seed').value);
      if(!Number.isInteger(settings.random_frames)||settings.random_frames<1||settings.random_frames>9000||!Number.isInteger(settings.random_seed)||settings.random_seed<0||settings.random_seed>4294967295)throw new Error('随机帧数请填写 1～9000 的整数，随机种子请填写 0～4294967295 的整数。');
    }
    // Older servers cap all-mode at 2400; selecting all 9000 candidates is equivalent.
    if(settings.frame_mode==='all'&&!nativeAllFrames){settings.frame_mode='random';settings.random_frames=9000;settings.random_seed=42;}
    const useSample=sample, entries=useSample?[{file:null,requestId:newId()}]:[...pendingFiles], batchId=entries.find(x=>x.batchId)?.batchId||newId();
    const clean=$('reference-clean').files[0], marked=$('reference-marked').files[0];
    uploading=true;$('run').disabled=true;$('video').disabled=true;$('use-sample').disabled=true;renderFiles();
    batchItems=entries.map(entry=>({name:entry.file?.name||'VideoSeal 已知水印样本.mp4',requestId:entry.requestId}));rememberBatch();
    let submitted=0;
    for(let index=0;index<entries.length;index++){
      const entry=entries[index],item=batchItems[index];entry.batchId=batchId;item.uploading=true;renderBatch();$('run').textContent=`正在上传 ${index+1} / ${entries.length}…`;
      try{
        if(entry.file&&(!/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(entry.file.name)||entry.file.size===0||entry.file.size>512*1024*1024))throw new Error('请选择非空视频文件，支持 MP4 / MOV / MKV / WEBM / AVI / M4V，单个最大 512 MB。');
        const form=new FormData();form.append('config',JSON.stringify(settings));form.append('sample',String(useSample));form.append('batch_id',batchId);form.append('request_id',entry.requestId);if(entry.file)form.append('video',entry.file);if(settings.methods.includes('carrier')){form.append('reference_clean',clean);form.append('reference_marked',marked);}
        let job;
        try{job=await api('/api/jobs',{method:'POST',body:form});}catch(error){
          // Query before retrying a possibly accepted upload; the request ID also deduplicates retries.
          try{const latest=await api('/api/jobs');job=latest.find(x=>x.request_id===entry.requestId);}catch{}
          if(!job)throw error;
        }
        item.jobId=job.id;submitted++;pendingFiles=pendingFiles.filter(x=>x!==entry);
        jobs=[job,...jobs.filter(x=>x.id!==job.id)];
        if(submitted===1){activeId=job.id;localStorage.setItem('spectral-job',activeId);renderedResult=null;renderJob(job);}
      }catch(error){item.error=error.message;}finally{item.uploading=false;rememberBatch();renderBatch();}
    }
    await poll();if(submitted<entries.length){$('form-error').textContent=`已提交 ${submitted} 个，${entries.length-submitted} 个未提交。未提交文件保留在选择列表中，可检查原因后重试。`;$('form-error').hidden=false;}toast(`已提交 ${submitted} 个任务，按顺序串行处理。`);
  }catch(error){$('form-error').textContent=error.message;$('form-error').hidden=false;}finally{uploading=false;$('run').disabled=false;$('video').disabled=false;$('use-sample').disabled=false;$('run').innerHTML='开始实验 <span>→</span>';renderFiles();renderBatch();}
};
function rememberBatch(){localStorage.setItem('spectral-batch',JSON.stringify(batchItems.map(({uploading,...item})=>item)));}
function renderBatch(){
  $('batch-panel').hidden=!batchItems.length;if(!batchItems.length)return;
  const rows=batchItems.map(item=>({item:{...item,error:item.error?friendlyError(item.error):''},job:jobs.find(j=>j.id===item.jobId||j.request_id&&j.request_id===item.requestId)}));
  const completed=rows.filter(x=>x.job?.status==='completed').length,failed=rows.filter(x=>x.item.error||['failed','cancelled','interrupted'].includes(x.job?.status)).length,active=rows.filter(x=>x.job&&running(x.job.status)).length;
  $('batch-summary').textContent=`共 ${rows.length} 个 · 已完成 ${completed} · 待处理 ${active} · 异常或取消 ${failed}${uploading?' · 正在上传，请保持页面打开':''}`;
  $('cancel-batch').hidden=!active;$('cancel-batch').disabled=uploading;
  $('batch-list').innerHTML=rows.map(({item,job},index)=>`<div class="batch-row"><div class="batch-main"><strong>${index+1}. ${escapeHtml(item.name)}</strong><small>${escapeHtml(job?job.phase:item.uploading?'正在上传…':item.error||(uploading?'等待上传':'尚未提交；刷新后请重新选择文件'))}${job&&running(job.status)?` · ${job.progress}%`:''}</small></div><span class="status-pill ${job?.status||(item.error?'failed':'neutral')}">${escapeHtml(job?states[job.status]:item.uploading?'上传中':item.error?'提交失败':'待上传')}</span>${job?`<button class="text-button" data-batch-job="${job.id}">查看</button>`:''}${job?.status==='completed'?`<a class="secondary-button" href="${fileUrl(job.id,'result.mp4',true)}">下载视频</a>`:''}</div>`).join('');
}
function showJob(id){activeId=id;localStorage.setItem('spectral-job',id);renderedResult=null;switchView('workbench');const job=jobs.find(j=>j.id===id);if(job){renderJob(job);if(job.status==='completed')$('results').scrollIntoView({behavior:'smooth',block:'start'});}}
$('batch-list').onclick=event=>{const button=event.target.closest('[data-batch-job]');if(button)showJob(button.dataset.batchJob);};
$('cancel-batch').onclick=async()=>{const active=jobs.filter(j=>batchItems.some(x=>x.jobId===j.id||x.requestId===j.request_id)&&running(j.status));$('cancel-batch').disabled=true;try{for(const job of active)await api(`/api/jobs/${job.id}/cancel`,{method:'POST'});await poll();}catch(error){toast(error.message);}finally{$('cancel-batch').disabled=false;}};
function renderJob(job){
  $('empty-task').hidden=true;$('active-task').hidden=false;$('job-name').textContent=job.name;$('job-status').textContent=states[job.status]||job.status;$('job-status').className=`status-pill ${job.status}`;$('job-phase').textContent=job.phase;$('job-percent').textContent=`${job.progress}%`;$('job-bar').style.width=`${job.progress}%`;$('job-elapsed').textContent=`耗时 ${running(job.status)&&job.started_at?Math.max(0,Math.round((Date.now()-new Date(job.started_at).getTime())/1000)):job.elapsed_seconds||0} 秒`;$('cancel').hidden=!running(job.status);$('cancel').disabled=job.status==='cancelling';$('job-logs').innerHTML=(job.logs||[]).map(l=>`<div class="log-item"><span class="log-time">${escapeHtml(l.time)}</span>${escapeHtml(l.text)}</div>`).join('');$('job-error').textContent=job.error||'';$('job-error').hidden=!job.error;
  if(job.status==='completed'&&job.id!==renderedResult)renderResults(job);
  if(job.status!=='completed')$('results').hidden=true;
}
const fileUrl=(id,name,download=false)=>`/api/jobs/${id}/files/${name}${download?'?download=true':''}`;
function renderResults(job){
  const report=job.report, m=report.metrics;renderedResult=job.id;$('results').hidden=false;$('result-scope').textContent=m.watermark_note+(report.acceleration?` 实际计算：${report.acceleration.device_name}；编码：${report.acceleration.encoder}。`:'');
  if(report.acceleration?.vae?.strategy){const tile=report.acceleration.vae.tile_size;$('result-scope').textContent+=` VAE：${tile?`${tile} px 分块`:'整帧计算'}。`;}
  if(report.acceleration?.denoise?.backend)$('result-scope').textContent+=` 去噪：${report.acceleration.denoise.backend==='opencl'?'OpenCL GPU':'CPU'}。`;
  if(report.sampling&&report.sampling.selected_frames===report.sampling.total_frames)$('result-scope').textContent+=` 所选处理片段全部 ${report.sampling.total_frames} 帧执行算法（100%）。`;
  else if(report.sampling?.mode==='random')$('result-scope').textContent+=` 随机抽帧：${report.sampling.total_frames} 帧中处理 ${report.sampling.selected_frames} 帧（${report.sampling.selected_percent}%），种子 ${report.sampling.seed}。对照图片取第 ${report.preview_frame_number} 帧。画质和消息指标仍仅抽样前 10 秒，可能未覆盖随机处理帧。`;
  if(report.export?.mode==='full')$('result-scope').textContent+=` 完整导出：${report.export.output_seconds.toFixed(2)} 秒，其中处理片段 ${report.export.processed_seconds.toFixed(2)} 秒，后段 ${report.export.tail_seconds.toFixed(2)} 秒不执行算法但仍统一编码。左侧对照预览仅为处理片段。`;
  const bit=m.videoseal_after===undefined?'未验证':`${(m.videoseal_after*100).toFixed(2)}%`;
  const bitNote=m.videoseal_before===undefined?'未知平台水印无可靠判定':`处理前 ${(m.videoseal_before*100).toFixed(2)}% · 考虑比特反转后 ${(m.complement_aware_after*100).toFixed(2)}%`;
  const frameNote=report.sampling?.mode==='random'?`输出 ${report.processed_frames} 帧 / 算法处理 ${report.sampling.selected_frames} 帧`:`${report.processed_frames} 帧`;
  const metrics=[['抽样 PSNR',m.sample_psnr_db===null?'∞':`${m.sample_psnr_db.toFixed(2)} dB`,'对照副本与结果解码比较'],['抽样 SSIM',m.sample_ssim.toFixed(4),`${m.sample_frames} 个对齐抽样帧`],['VideoSeal 消息恢复率',bit,bitNote],['处理耗时',`${job.elapsed_seconds} s`,`${frameNote} · ${report.processing_resolution.join(' × ')}`]];
  $('metrics').innerHTML=metrics.map(([label,value,note])=>`<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></div>`).join('');
  $('before-video').poster=fileUrl(job.id,'before.png');$('after-video').poster=fileUrl(job.id,'after.png');$('before-video').src=fileUrl(job.id,'before.mp4');$('after-video').src=fileUrl(job.id,'result.mp4');$('download-video').href=fileUrl(job.id,'result.mp4',true);$('download-report').href=fileUrl(job.id,'report.json',true);for(const[id,name]of [['before-image','before.png'],['after-image','after.png'],['difference-image','difference.png']])$(id).src=fileUrl(job.id,name);
}
$('cancel').onclick=async()=>{if(activeId)try{renderJob(await api(`/api/jobs/${activeId}/cancel`,{method:'POST'}));}catch(error){toast(error.message);}};
$('sync-video').onclick=async()=>{const a=$('before-video'),b=$('after-video');if(!a.paused||!b.paused){a.pause();b.pause();$('sync-video').textContent='同步播放';return;}a.currentTime=0;b.currentTime=0;b.muted=true;try{await Promise.all([a.play(),b.play()]);$('sync-video').textContent='暂停播放';}catch{toast('浏览器未允许同步播放，请使用各视频的播放按钮。');}};
$('after-video').addEventListener('ended',()=>{$('sync-video').textContent='同步播放';});
function renderHistory(){
  $('history-count').textContent=jobs.length;$('history-list').innerHTML=jobs.length?jobs.map(job=>`<div class="history-row"><div class="history-main"><strong>${escapeHtml(job.name)}</strong><small>${new Date(job.created_at).toLocaleString('zh-CN')} · ${job.config.methods.map(id=>escapeHtml(methods.find(m=>m.id===id)?.name||id)).join(' → ')}</small></div><span class="status-pill ${job.status}">${states[job.status]}</span><button class="secondary-button" data-job="${job.id}">查看 ${job.status==='completed'?'结果':'进度'} →</button></div>`).join(''):'<div class="history-empty">还没有实验记录。从验证样本开始吧。</div>';
}
$('history-list').onclick=event=>{const button=event.target.closest('[data-job]');if(button)showJob(button.dataset.job);};
function switchView(view){$('workbench-view').hidden=view!=='workbench';$('history-view').hidden=view!=='history';$('page-title').innerHTML=view==='history'?'任务与记录<span class="title-dot">.</span>':'视频水印实验台<span class="title-dot">.</span>';$('page-label').textContent=view==='history'?'任务与记录':'实验工作台';document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));if(view==='history')$('results').hidden=true;else if(renderedResult===activeId)$('results').hidden=false;}
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>switchView(button.dataset.view));
async function poll(){try{jobs=await api('/api/jobs');renderHistory();renderBatch();const active=jobs.find(j=>j.id===activeId);if(active)renderJob(active);}catch{if(!poll.warned){toast('与本地服务的连接已中断，记录保存在磁盘。');poll.warned=true;}}}
async function init(){await loadCapabilities();await poll();setInterval(poll,1800);}init();
