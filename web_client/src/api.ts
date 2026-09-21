export async function api<T>(path:string, init?:RequestInit):Promise<T>{
  const response=await fetch(path,{...init,headers:{'Content-Type':'application/json',...(init?.headers||{})}});
  const body=await response.json();
  if(!response.ok) throw new Error(body.detail||body.error||`Request failed (${response.status})`);
  return body as T;
}
export function download(name:string,value:unknown){
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
  const link=document.createElement('a'); link.href=url; link.download=name; link.click(); URL.revokeObjectURL(url);
}
export async function filesBase64(files:FileList){
  const result:Record<string,string>={};
  for(const file of Array.from(files)){const bytes=new Uint8Array(await file.arrayBuffer());let raw='';bytes.forEach(x=>raw+=String.fromCharCode(x));result[file.name]=btoa(raw)}
  return result;
}
