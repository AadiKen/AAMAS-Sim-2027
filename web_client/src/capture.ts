import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import runtime from '../../presentation_assets/screenshots/hero_runtime_state.json';
import surveyorUrl from '../../presentation_assets/screenshots/surveyor_v4_nominal.glb?url';

const mode = new URLSearchParams(location.search).get('mode') || 'hero';
const scene = new THREE.Scene();
scene.background = new THREE.Color('#071c29');
const camera = new THREE.PerspectiveCamera(39, innerWidth / innerHeight, 0.1, 500);
const renderer = new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setSize(innerWidth,innerHeight);
renderer.setPixelRatio(Math.min(2,window.devicePixelRatio));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = mode === 'mesh' ? 1.45 : 1.55;
document.body.appendChild(renderer.domElement);
scene.add(new THREE.HemisphereLight(mode === 'mesh' ? 0xffffff : 0xc4e8f4, 0x234052, 2.4));
const sun = new THREE.DirectionalLight(0xffffff,3.3); sun.position.set(12,30,18); scene.add(sun);
const mat = (color:string, metalness=0, roughness=.65) => new THREE.MeshStandardMaterial({color,metalness,roughness,side:THREE.DoubleSide});
function caption(title:string,subtitle:string) { document.getElementById('caption')!.innerHTML = `${title}<span id="sub">${subtitle}</span>`; }
function water(size:number) {
 const plane = new THREE.Mesh(new THREE.PlaneGeometry(size,size,80,80),mat('#176078',.24,.46));
 plane.rotation.x=-Math.PI/2;plane.position.y=-.18;scene.add(plane);
 const grid = new THREE.GridHelper(size,20,'#4e9eac','#328297');grid.position.y=-.17;grid.material.transparent=true;grid.material.opacity=.22;scene.add(grid);
}
function boat(x:number,y:number,heading:number,color:string) {
 const g=new THREE.Group();g.position.set(x,.17,-y);g.rotation.y=heading;
 const sh=new THREE.Shape();sh.moveTo(2.2,0);sh.lineTo(.85,.63);sh.lineTo(-1.8,.63);sh.lineTo(-2.15,.35);sh.lineTo(-2.15,-.35);sh.lineTo(-1.8,-.63);sh.lineTo(.85,-.63);sh.closePath();
 const hull=new THREE.Mesh(new THREE.ExtrudeGeometry(sh,{depth:.43,bevelEnabled:true,bevelThickness:.07,bevelSize:.09,bevelSegments:2}),mat(color,.2,.35));
 hull.rotation.x=-Math.PI/2;g.add(hull);
 const deck=new THREE.Mesh(new THREE.BoxGeometry(1.05,.32,.7),mat('#e6eeeb',.12,.38));deck.position.set(-.25,.62,0);g.add(deck);
 const window=new THREE.Mesh(new THREE.BoxGeometry(.46,.19,.74),mat('#15384e',.12,.2));window.position.set(.1,.78,0);g.add(window);
 scene.add(g);
 const ring=new THREE.Mesh(new THREE.RingGeometry(1.36,1.41,64),new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide,transparent:true,opacity:.56}));ring.rotation.x=-Math.PI/2;ring.position.set(x,-.14,-y);scene.add(ring);
}
function obstacle(x:number,y:number,r:number) {
 const base = new THREE.Mesh(new THREE.CylinderGeometry(r,r,.35,40),mat('#d1b183',.05,.9));base.position.set(x,.08,-y);scene.add(base);
 const top = new THREE.Mesh(new THREE.CylinderGeometry(r*.78,r*.92,.7,40),mat('#718a75',.05,.95));top.position.set(x,.43,-y);scene.add(top);
}
function hero() {
 water(90);
 runtime.obstacles.forEach(o=>obstacle(o.x_m,o.y_m,o.radius_m));
 ['#36d1c4','#f5c768','#eb8b66','#8eacd8'].forEach((c,i)=>{
  const v=runtime.vessels[i];boat(v.x_m,v.y_m,v.heading_rad,c);
 });
 camera.position.set(0,61,69);camera.lookAt(0,0,0);
 caption('MANTA / FOUR-VESSEL SIMULATION','Full6 Otter scene · t = 2.0 s · actual simulator state · vessel shapes are display proxies');
 renderer.render(scene,camera);
}
function meshView(mesh:THREE.Object3D, inRuntime:boolean) {
 mesh.rotation.x=-Math.PI/2;
 const b=new THREE.Box3().setFromObject(mesh), size=b.getSize(new THREE.Vector3()), center=b.getCenter(new THREE.Vector3());
 mesh.position.sub(center);mesh.position.y+=inRuntime ? .05 : 0;
 mesh.traverse((o:any)=>{if(o.isMesh)o.material=new THREE.MeshBasicMaterial({color:inRuntime?'#dfc88f':'#72bfc5',side:THREE.DoubleSide});});
 if(inRuntime){
  water(8);
  // A frozen V01-USV1 replay pose supplies the heading; the view recenters on that vessel.
  mesh.rotation.y=2.117084382669122;
  const ring=new THREE.Mesh(new THREE.RingGeometry(1.15,1.18,60),new THREE.MeshBasicMaterial({color:'#cfe4df',side:THREE.DoubleSide,transparent:true,opacity:.6}));
  ring.rotation.x=-Math.PI/2;ring.position.y=-.14;scene.add(ring);
  camera.position.set(3.3,3.4,4.5);camera.lookAt(0,.05,0);
  caption('SURVEYOR / MANTA REPLAY','Nominal v4 repaired hull at frozen V01-USV1 initial simulated pose · geometry display, not native photorealistic rendering');
 } else {
  camera.position.set(3.2,2.45,3.65);camera.lookAt(0,0,0);
  caption('SEAROBOTICS SURVEYOR','Source-constrained repaired nominal v4 hull mesh · watertight passive-package input');
 }
 scene.add(mesh);renderer.render(scene,camera);
}
if(mode==='hero') hero();
else new GLTFLoader().load(surveyorUrl,gltf=>meshView(gltf.scene,mode==='runtime'));
