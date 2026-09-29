"""Make reproducible source-face views and section overlays for pocket review."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def run(root: Path, mirror_path: Path) -> None:
    inventory = json.loads((root/"face_inventory.json").read_text())["faces"]
    packed = np.load(root/"face_triangles.npz")
    render = root/"renders"; (render/"sections").mkdir(parents=True, exist_ok=True)
    mirror_all = trimesh.load_mesh(mirror_path, process=True)
    mirror = next(p for p in mirror_all.split(only_watertight=False) if p.bounds[0,1]<0)
    ghost = np.column_stack((-mirror.vertices[:,1]*1000,-mirror.vertices[:,2]*1000,
                              mirror.vertices[:,0]*1000+915))
    top = sorted(inventory, key=lambda r:-r["pocket_patch_area_mm2"])[:30]
    major_ids = {r["face_id"] for r in top}
    cmap = plt.get_cmap("turbo")
    color = {r["face_id"]: cmap(i/max(1,len(inventory)-1)) for i,r in enumerate(inventory)}
    patches = {}
    for row in inventory:
        sid=row["face_id"]
        v=packed[sid+"_vertices"];f=packed[sid+"_triangles"]
        m=trimesh.Trimesh(vertices=v,faces=f,process=False)
        c=m.triangles_center
        q=(c[:,0]>0)&(c[:,2]>=900)&(c[:,2]<=1200)&(c[:,1]<=0)
        patches[sid]=c[q]
    def scatter_view(name: str, horizontal: int, vertical: int, title: str,
                     invert_horizontal: bool=False, sample: int=1500):
        fig, ax=plt.subplots(figsize=(12,8))
        g=ghost[::max(1,len(ghost)//9000)]
        ax.scatter(g[:,horizontal],g[:,vertical],s=.13,c="0.75",alpha=.18,
                   rasterized=True,label="mirrored port envelope")
        for row in inventory:
            sid=row["face_id"];p=patches[sid]
            if len(p)==0:continue
            p=p[::max(1,len(p)//sample)]
            ax.scatter(p[:,horizontal],p[:,vertical],s=2.5,c=[color[sid]],alpha=.85,
                       rasterized=True)
            if sid in major_ids and (row["mirror_vertex_distance_p95_mm"]>5 or row["pocket_patch_area_mm2"]>10000):
                mean=p.mean(axis=0)
                ax.text(mean[horizontal],mean[vertical],sid,fontsize=7,color="black",
                        bbox=dict(facecolor="white",alpha=.72,edgecolor="none",pad=.6))
        if vertical==1:ax.axhline(-78.2695,color="navy",ls="--",lw=1.5,label="52.3 kg waterline")
        if horizontal==1:ax.axvline(-78.2695,color="navy",ls="--",lw=1.5,label="52.3 kg waterline")
        ax.set(xlabel=["source X (mm)","source Y (mm)","source Z (mm)"][horizontal],
               ylabel=["source X (mm)","source Y (mm)","source Z (mm)"][vertical],title=title)
        if invert_horizontal:ax.invert_xaxis()
        ax.grid(alpha=.2);ax.legend(loc="best",fontsize=8)
        fig.tight_layout();fig.savefig(render/name,dpi=170);plt.close(fig)
    scatter_view("face_ids_external.png",2,1,"Starboard side: original face IDs over mirrored envelope")
    scatter_view("face_ids_internal.png",2,1,"Starboard side, viewed from inside",True)
    scatter_view("stern.png",0,1,"Transverse pocket face evidence")
    scatter_view("underside.png",2,0,"Pocket underside: source face IDs")
    scatter_view("mirrored_overlay.png",0,2,"Original starboard faces and mirrored port, plan view")
    # 3D perspective uses a deterministic subsample, with high-distance faces
    # identified by labels on the more legible orthographic views.
    fig=plt.figure(figsize=(12,9));ax=fig.add_subplot(111,projection="3d")
    for row in inventory:
        sid=row["face_id"];p=patches[sid]
        if len(p)==0:continue
        p=p[::max(1,len(p)//400)]
        ax.scatter(p[:,2],p[:,0],p[:,1],s=1,c=[color[sid]],rasterized=True)
    g=ghost[::max(1,len(ghost)//5000)]
    ax.scatter(g[:,2],g[:,0],g[:,1],s=.1,c="0.65",alpha=.12)
    ax.set(xlabel="source Z mm",ylabel="source X mm",zlabel="source Y mm",
           title="Original starboard pocket faces by stable face ID")
    ax.view_init(elev=22,azim=-55);fig.tight_layout();fig.savefig(render/"perspective.png",dpi=170);plt.close(fig)
    fig=plt.figure(figsize=(12,9));ax=fig.add_subplot(111,projection="3d")
    for row in inventory:
        sid=row["face_id"];p=patches[sid]
        if len(p)==0:continue
        p=p[::max(1,len(p)//400)].copy()
        is_cone=sid in ("SF057","SF058")
        if is_cone:p[:,0]+=110
        ax.scatter(p[:,2],p[:,0],p[:,1],s=1,
                   c=["tab:red" if is_cone else "tab:blue"],alpha=.8,rasterized=True)
    ax.set(xlabel="source Z mm",ylabel="source X mm (+110 for two cone faces)",
           zlabel="source Y mm",
           title="Analytical exploded view: cone faces are connected to the same CAD group")
    ax.view_init(elev=22,azim=-55);fig.tight_layout();fig.savefig(render/"exploded_feature.png",dpi=170);plt.close(fig)
    # Source-face triangles retain tags for section cuts. These diagrams show
    # original curves, without pretending they form an exterior closed ring.
    all_source=trimesh.util.concatenate([
        trimesh.Trimesh(vertices=packed[r["face_id"]+"_vertices"],
                        faces=packed[r["face_id"]+"_triangles"],process=False)
        for r in inventory])
    for station in (975,1025,1075,1125,1175):
        fig,ax=plt.subplots(figsize=(7,7))
        for row in inventory:
            sid=row["face_id"]
            mesh=trimesh.Trimesh(vertices=packed[sid+"_vertices"],
                                faces=packed[sid+"_triangles"],process=False)
            seg,_=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,station],return_faces=True)
            if len(seg):
                for segment in seg:
                    ax.plot(segment[:,0],segment[:,1],color=color[sid],lw=.8)
        gseg,_=trimesh.intersections.mesh_plane(trimesh.Trimesh(vertices=ghost,faces=mirror.faces,process=False),
                                                [0,0,1],[0,0,station],return_faces=True)
        for segment in gseg:ax.plot(segment[:,0],segment[:,1],color="black",lw=.6,alpha=.6)
        ax.axhline(-78.2695,color="navy",ls="--",lw=1)
        ax.set(xlabel="source X mm",ylabel="source Y mm",title=f"Transverse Z={station} mm",
               xlim=(230,445),ylim=(-255,30),aspect="equal")
        fig.tight_layout();fig.savefig(render/"sections"/f"transverse_{station}.png",dpi=160);plt.close(fig)
    for lateral in (275,338,400):
        fig,ax=plt.subplots(figsize=(10,5))
        seg,_=trimesh.intersections.mesh_plane(all_source,[1,0,0],[lateral,0,0],return_faces=True)
        for segment in seg:ax.plot(segment[:,2],segment[:,1],color="tab:orange",lw=.7)
        gseg,_=trimesh.intersections.mesh_plane(trimesh.Trimesh(vertices=ghost,faces=mirror.faces,process=False),
                                                [1,0,0],[lateral,0,0],return_faces=True)
        for segment in gseg:ax.plot(segment[:,2],segment[:,1],color="black",lw=.6)
        ax.axhline(-78.2695,color="navy",ls="--",lw=1)
        ax.set(xlabel="source Z mm",ylabel="source Y mm",title=f"Longitudinal X={lateral} mm",
               xlim=(880,1220),ylim=(-255,30))
        fig.tight_layout();fig.savefig(render/"sections"/f"longitudinal_{lateral}.png",dpi=160);plt.close(fig)
    print(f"rendered {len(list(render.rglob('*.png')))} evidence views")


if __name__ == "__main__":
    if len(sys.argv)!=3:raise SystemExit("usage: python tools/render_surveyor_pocket_evidence.py ROOT MIRRORED_STL")
    run(Path(sys.argv[1]),Path(sys.argv[2]))
