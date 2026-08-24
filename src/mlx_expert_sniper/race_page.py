"""Built-in Fast Token race page, served same-origin by the Ollama server
at /race — no CORS, no mixed-content, works in every browser."""

RACE_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fast Token race</title><style>
:root{--bg:#08070C;--card:#14101F;--ink:#EDEAFB;--mut:#7C7494;--line:#241E36;--acc:#8F7CF6}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,Helvetica,sans-serif;padding:24px;max-width:960px;margin:0 auto}
h1{font-size:26px;margin:6px 0 4px}.mut{color:var(--mut);font-size:13px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:760px){.row{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.out{height:250px;overflow-y:auto;white-space:pre-wrap;background:#0D0B14;
border:1px solid var(--line);border-radius:8px;padding:10px;
font:12px/1.6 ui-monospace,Menlo,monospace;margin-top:8px}
.meter{font:12px ui-monospace,Menlo,monospace;color:var(--mut);margin-top:8px}
.meter b{color:var(--acc)}
button{background:var(--acc);color:#0D0B14;border:0;border-radius:999px;
padding:9px 20px;font-weight:700;font-size:14px;cursor:pointer;margin-top:12px}
button:disabled{opacity:.5}input{width:100%;background:#0D0B14;color:var(--ink);
border:1px solid var(--line);border-radius:8px;padding:8px;font:12px ui-monospace,monospace;margin-top:6px}
.win{color:var(--acc);font:700 16px ui-monospace,monospace;margin-left:14px}
.lane-t{font-weight:700;font-size:14px}</style></head><body>
<div class="mut">expert sniper &middot; local demo</div>
<h1>Same model, same prompt, twice</h1>
<p class="mut">Round one decodes normally. Round two turns on Fast Token: a draft
node proposes tokens, this model verifies the batch in one pass. Back to back on
this machine's engine, which is the honest side-by-side.</p>
<p class="mut" style="margin-top:6px">What to expect: on a machine that streams
experts from SSD, watch the right lane's meters &mdash; drafts accepted live,
several tokens per forward &mdash; but the multiplier usually lands at or below
1&times;, because the verify batch reads more experts than it saves. The
speedup is real where experts sit resident in memory: network nodes, or a
32&nbsp;GB+ machine whose cache holds the whole model. Every number on screen
is measured, nothing is staged.</p>
<div class="card" style="margin-top:14px">
<div class="mut">Prompt</div>
<input id="prompt" value="Write a Python function that reverses a linked list, with a docstring and a short example.">
<div class="row" style="margin-top:8px">
<div><div class="mut">Draft node</div><input id="draft" value="http://66.94.126.39:8312/v1"></div>
<div><div class="mut">Draft tokens per round (K)</div><input id="k" value="4"></div>
</div>
<button id="go">Race</button><span class="win" id="win"></span>
<div class="mut" id="status" style="margin-top:6px"></div></div>
<div class="row">
<div class="card"><span class="lane-t">&#128034; Standard decode</span>
<div class="out" id="out-std">waiting&hellip;</div><div class="meter" id="m-std"></div></div>
<div class="card"><span class="lane-t" style="color:var(--acc)">&#9889; Fast Token</span>
<div class="out" id="out-fast">waiting&hellip;</div><div class="meter" id="m-fast"></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
let tps={};
async function lane(key,opts){
  const out=$("out-"+key),met=$("m-"+key);out.textContent="";met.textContent="";
  const t0=performance.now();let ttft=null,tok=0,spec=null;
  const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({messages:[{role:"user",content:$("prompt").value}],stream:true,
      options:Object.assign({num_predict:120},opts)})});
  const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
  for(;;){const{done,value}=await rd.read();if(done)break;
    buf+=dec.decode(value,{stream:true});const ls=buf.split("\\n");buf=ls.pop();
    for(const l of ls){if(!l.trim())continue;let j;try{j=JSON.parse(l)}catch{continue}
      if(j.done){spec=j.spec||null;continue}
      const c=j.message&&j.message.content;if(!c)continue;
      if(ttft===null)ttft=(performance.now()-t0)/1e3;
      tok++;out.textContent+=c;out.scrollTop=out.scrollHeight;
      const ds=(performance.now()-t0)/1e3-ttft;
      tps[key]=ds>.3?tok/ds:null;
      met.innerHTML="<b>"+(tps[key]?tps[key].toFixed(2)+" tok/s":"&mdash;")+
        "</b> &middot; TTFT "+ttft.toFixed(1)+"s &middot; "+tok+" tok";}}
  if(spec)met.innerHTML+=" &middot; "+(tok/spec.forwards).toFixed(1)+
    " tok/forward &middot; "+Math.round(100*spec.accepted/Math.max(1,spec.drafted))+"% drafts accepted";
}
$("go").onclick=async()=>{
  $("go").disabled=true;$("win").textContent="";tps={};
  try{
    $("status").textContent="Round 1: standard decode\\u2026";
    await lane("std",{});
    $("status").textContent="Round 2: Fast Token\\u2026";
    await lane("fast",{spec:true,draft_url:$("draft").value,spec_k:parseInt($("k").value)||4});
    $("status").textContent="done";
    if(tps.std&&tps.fast)$("win").textContent=(tps.fast/tps.std).toFixed(2)+"\\u00d7 with Fast Token";
  }catch(e){$("status").textContent="failed: "+e.message}
  $("go").disabled=false;};
</script></body></html>"""
