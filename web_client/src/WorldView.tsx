import {Canvas} from '@react-three/fiber';
import {useMemo} from 'react';
import {BufferGeometry, Line, LineBasicMaterial, Quaternion, Vector3} from 'three';
import {bodyQuaternion,frdToLocal,mountedDirection,nedToWeb,rpyToQuat,type Quat,type Vec3} from './spatial';
import {POSITION_NED,type ReplayState} from './contracts';

type Definition={kind:string;id:string;version:string;payload:Record<string,any>};
type Frame={states:Record<string,ReplayState>};
type Project={config:any;definitions:Definition[]};
type TrailFrame={states:Record<string,{position_display_m:Vec3}>};
type Props={project:Project;frame?:Frame|null;trail?:TrailFrame[];selected?:string;onSelect?:(id:string)=>void;onMoveGoal?:(ned:Vec3)=>void};
const asVec=(value:unknown,fallback:Vec3=[0,0,0]):Vec3=>Array.isArray(value)&&value.length===3?value as Vec3:fallback;
const asQuat=(value:unknown):Quat=>Array.isArray(value)&&value.length===4?value as Quat:[1,0,0,0];
function Marker({position,color,onClick}:{position:Vec3;color:string;onClick?:()=>void}){return <mesh position={position} onClick={e=>{e.stopPropagation();onClick?.()}}><sphereGeometry args={[.2,12,8]}/><meshStandardMaterial color={color}/></mesh>}
function Arrow({position,direction,color,length=1}:{position:Vec3;direction:Vec3;color:string;length?:number}){
 const dir=new Vector3(...direction).normalize();return <arrowHelper args={[dir,new Vector3(...position),length,color,.25,.12]}/>;
}
export function WorldView({project,frame,trail=[],selected,onSelect,onMoveGoal}:Props){
 const config=project.config||{},vessel=config.vessels?.[0],world=config.world||{};
 const vesselDef=project.definitions.find(d=>d.kind==='vessel'&&`${d.id}@${d.version}`===vessel?.definition);
 const taskDef=project.definitions.find(d=>d.kind==='task'&&`${d.id}@${d.version}`===config.task?.type);
 const sensorDefs=project.definitions.filter(d=>d.kind==='sensor'&&vessel?.sensors?.includes(`${d.id}@${d.version}`));
 const actuatorDefs=project.definitions.filter(d=>d.kind==='actuator'&&vessel?.actuators?.includes(`${d.id}@${d.version}`));
 const state=frame?.states?.[vessel?.instance_id],spawn=asVec(vessel?.spawn?.ned_m);
 const pose=state?.position_display_m||nedToWeb(spawn), q=state?.q_body_to_ned||rpyToQuat(asVec(vessel?.spawn?.rpy_rad));
 const orientation=useMemo(()=>bodyQuaternion(q),[q[0],q[1],q[2],q[3]]);
 const half=asVec(vesselDef?.payload?.collision?.half_extents_m,[1.2,.45,.3]);
 const goal=asVec(taskDef?.payload?.target_ned_m,[8,0,0]);
 const current=asVec(world.environment?.current?.ned_mps),wind=asVec(world.environment?.wind?.ned_mps);
 const line=useMemo(()=>{const points=trail.map(f=>f.states?.[vessel?.instance_id]?.position_display_m).filter(Boolean).map(p=>new Vector3(...p as Vec3));return new BufferGeometry().setFromPoints(points)},[trail,vessel?.instance_id]);
 const trajectory=useMemo(()=>new Line(line,new LineBasicMaterial({color:'#f6e17c'})),[line]);
 return <div className="viewport"><Canvas camera={{position:[13,12,19],fov:45}}><color attach="background" args={['#06141b']}/><ambientLight intensity={1.4}/><directionalLight position={[5,10,4]} intensity={2}/>
  <gridHelper args={[40,40,'#28677b','#123541']}/>
  <mesh rotation={[-Math.PI/2,0,0]} position={[0,-.03,0]} onClick={e=>{if(onMoveGoal){e.stopPropagation();onMoveGoal([e.point.z,e.point.x,goal[2]])}}}><planeGeometry args={[100,100]}/><meshBasicMaterial transparent opacity={0}/></mesh>
  <group position={pose} quaternion={orientation}>
   <mesh onClick={e=>{e.stopPropagation();onSelect?.('vessel')}}><boxGeometry args={[2*half[1],2*half[2],2*half[0]]}/><meshStandardMaterial color={selected==='vessel'?'#74f9d9':'#36d7b7'} metalness={.15} roughness={.4}/></mesh>
   <Arrow position={[0,0,0]} direction={[0,0,1]} color="#fbce6b" length={2}/>
   {actuatorDefs.map(d=>{const p=frdToLocal(asVec(d.payload.mount_frd_m));const dir=mountedDirection(asQuat(d.payload.mount_q_to_frd),[1,0,0]);return <group key={d.id}><Marker position={p} color="#ff9b6f" onClick={()=>onSelect?.(`actuator:${d.id}`)}/><Arrow position={p} direction={dir} color="#ff9b6f" length={1.1}/></group>})}
   {sensorDefs.map(d=>{const p=frdToLocal(asVec(d.payload.mount_frd_m));const dir=mountedDirection(asQuat(d.payload.mount_q_to_frd),d.payload.kind==='sonar'?[0,0,1]:[1,0,0]);const unit=new Vector3(...dir).normalize();const coneQ=new Quaternion().setFromUnitVectors(new Vector3(0,1,0),unit.clone().negate());const len=d.payload.kind==='sonar'?Math.min(3,Number(d.payload.max_range_m)||3):0;return <group key={d.id}><Marker position={p} color="#8ac4ff" onClick={()=>onSelect?.(`sensor:${d.id}`)}/>{len>0&&<mesh position={new Vector3(...p).addScaledVector(unit,len/2)} quaternion={coneQ}><coneGeometry args={[len*Math.tan((Number(d.payload.fov_rad)||.3)/2),len,16,1,true]}/><meshBasicMaterial color="#5daefc" transparent opacity={.12} depthWrite={false}/></mesh>}</group>})}
  </group>
  <Marker position={nedToWeb(goal)} color="#ffdb7d" onClick={()=>onSelect?.('goal')}/>
  {(world.static_entities||[]).map((entity:any)=><Marker key={entity.id} position={nedToWeb(asVec(entity[POSITION_NED]))} color="#fa7272" onClick={()=>onSelect?.(`obstacle:${entity.id}`)}/>)}
  {current.some(x=>x!==0)&&<Arrow position={[3,.3,3]} direction={nedToWeb(current)} color="#62d8ff" length={2}/>}
  {wind.some(x=>x!==0)&&<Arrow position={[-3,.3,3]} direction={nedToWeb(wind)} color="#ded5ff" length={2}/>}
  {trail.length>1&&<primitive object={trajectory}/>}
 </Canvas><div className="viewport-key">Hull · amber heading · orange actuators · blue sensors · yellow goal · red obstacle</div></div>;
}
