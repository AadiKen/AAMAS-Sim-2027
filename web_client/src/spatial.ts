import {Matrix4, Quaternion, Vector3} from 'three';
export type Vec3=[number,number,number];
export type Quat=[number,number,number,number];
export const nedToWeb=([n,e,d]:Vec3):Vec3=>[e,-d,n];
export const webToNed=([e,u,n]:Vec3):Vec3=>[n,e,-u];
// Local scene axes are starboard, up, forward, matching the display axes at
// zero heading. Both transforms include the same reflection, so pose rotations
// are proper Three.js rotations.
export const frdToLocal=([f,r,d]:Vec3):Vec3=>[r,-d,f];
export function rpyToQuat([roll,pitch,yaw]:Vec3):Quat{
 const [cr,sr,cp,sp,cy,sy]=[Math.cos(roll/2),Math.sin(roll/2),Math.cos(pitch/2),Math.sin(pitch/2),Math.cos(yaw/2),Math.sin(yaw/2)];
 return [cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy];
}
export function quatToRpy([w,x,y,z]:Quat):Vec3{
 return [Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x)))),Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))];
}
export function rotate(q:Quat,v:Vec3):Vec3{
 const [w,x,y,z]=q,[vx,vy,vz]=v;
 const tx=2*(y*vz-z*vy),ty=2*(z*vx-x*vz),tz=2*(x*vy-y*vx);
 return [vx+w*tx+y*tz-z*ty,vy+w*ty+z*tx-x*tz,vz+w*tz+x*ty-y*tx];
}
export function bodyQuaternion(q:Quat):Quaternion{
 const right=nedToWeb(rotate(q,[0,1,0])),up=nedToWeb(rotate(q,[0,0,-1])),forward=nedToWeb(rotate(q,[1,0,0]));
 const matrix=new Matrix4().makeBasis(new Vector3(...right),new Vector3(...up),new Vector3(...forward));
 return new Quaternion().setFromRotationMatrix(matrix);
}
export function mountedDirection(q:Quat,axis:Vec3):Vec3{return frdToLocal(rotate(q,axis));}
