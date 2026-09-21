import { Canvas } from '@react-three/fiber';
type State={position_display_m:[number,number,number]};
export function WorldView({states}:{states:Record<string,State>}){
 return <div className="viewport"><Canvas camera={{position:[18,14,18],fov:45}}>
  <color attach="background" args={['#06141b']}/><ambientLight intensity={1.4}/><directionalLight position={[5,10,4]} intensity={2}/>
  <gridHelper args={[40,40,'#28677b','#123541']}/>
  {Object.entries(states).map(([id,state])=><group key={id} position={state.position_display_m}>
   <mesh><boxGeometry args={[2.4,.65,1.1]}/><meshStandardMaterial color="#36d7b7" metalness={.15} roughness={.4}/></mesh>
   <mesh position={[.8,.55,0]}><boxGeometry args={[.7,.5,.7]}/><meshStandardMaterial color="#e7f8f5"/></mesh>
  </group>)}
 </Canvas></div>
}
