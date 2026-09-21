import {useMemo,useState} from 'react';
import {api,download,filesBase64} from './api';
import {WorldView} from './WorldView';
import './styles.css';

const starter={config:{},definitions:[] as unknown[],actions:[] as unknown[]};
type Frame={master_step:number;sim_time_s:number;states:Record<string,{position_display_m:[number,number,number]}>;reward:Record<string,number>;terminated:boolean;termination_reason:string|null};
type Run={run_id:string;config_hash:string;frames:Frame[];artifact_path:string};
const steps=['Vessel','World','Scenario','Policy','Run'];

export default function App(){
 const [step,setStep]=useState(0),[text,setText]=useState(JSON.stringify(starter,null,2));
 const [status,setStatus]=useState('Ready'),[run,setRun]=useState<Run|null>(null),[frame,setFrame]=useState(0);
 const draft=useMemo(()=>{try{return JSON.parse(text)}catch{return null}},[text]);
 async function validate(){if(!draft)return setStatus('Fix the highlighted JSON first.');try{const r=await api<{config_hash:string}>('/v1/validate',{method:'POST',body:JSON.stringify({config:draft.config,definitions:draft.definitions})});setStatus(`Validated · ${r.config_hash.slice(0,12)}`)}catch(e){setStatus((e as Error).message)}}
 async function launch(){if(!draft)return;try{setStatus('Running on the backend…');const id=`web-${Date.now()}`;const value=await api<Run>('/v1/runs',{method:'POST',body:JSON.stringify({run_id:id,config:draft.config,definitions:draft.definitions,actions:draft.actions})});setRun(value);setFrame(value.frames.length-1);setStatus(`Run complete · ${value.frames.at(-1)?.termination_reason||'external stop'}`)}catch(e){setStatus((e as Error).message)}}
 async function upload(files:FileList|null){if(!files)return;try{const encoded=await filesBase64(files);const manifest=JSON.parse(atob(encoded['manifest.json']));await api('/v1/policies',{method:'POST',body:JSON.stringify({policy_id:manifest.policy_id,files_base64:encoded,observation_contract_hash:manifest.observation_contract_hash,action_contract_hash:manifest.action_contract_hash})});setStatus(`Policy ${manifest.policy_id} validated`)}catch(e){setStatus((e as Error).message)}}
 const shown=run?.frames[frame];
 return <main><header><div><span className="eyebrow">MARINE MULTI-AGENT RESEARCH</span><h1>BCOD <b>SIM</b></h1></div><div className="status"><i/> {status}</div></header>
  <div className="shell"><nav>{steps.map((name,i)=><button className={i===step?'active':''} onClick={()=>setStep(i)} key={name}><span>0{i+1}</span>{name}</button>)}</nav>
   <section className="work"><div className="title"><div><p>AUTHORING WORKFLOW</p><h2>{steps[step]}</h2></div><div className="actions"><button onClick={()=>draft&&download('bcod-experiment.json',draft)}>Export</button><button onClick={validate}>Validate</button><button className="primary" onClick={launch}>Run backend</button></div></div>
    {step===3?<div className="upload"><h3>Validated policy bundle</h3><p>Select the seven portable bundle files. Validation occurs on the server before the policy becomes available.</p><input type="file" multiple onChange={e=>upload(e.target.files)}/></div>:
    step===4?<><WorldView states={shown?.states||{}}/><div className="timeline"><input type="range" min="0" max={Math.max(0,(run?.frames.length||1)-1)} value={frame} onChange={e=>setFrame(Number(e.target.value))}/><span>Step {shown?.master_step||0} · {(shown?.sim_time_s||0).toFixed(2)} s</span></div><div className="cards"><article><label>Config</label><strong>{run?.config_hash.slice(0,12)||'—'}</strong></article><article><label>Reward</label><strong>{shown?Object.values(shown.reward).reduce((a,b)=>a+b,0).toFixed(3):'—'}</strong></article><article><label>Status</label><strong>{shown?.termination_reason||'Ready'}</strong></article></div></>:
    <div className="editor"><div className="hint"><h3>Canonical experiment</h3><p>Edit vessel definitions, mounted actuators and sensors, world fields, spawns, task rewards, and scenario actions. The server performs schema and physics validation.</p></div><textarea value={text} onChange={e=>setText(e.target.value)} spellCheck={false}/></div>}
   </section>
  </div><footer>Backend-authoritative simulation · NED / FRD / SI · Reproducible artifacts</footer>
 </main>
}
