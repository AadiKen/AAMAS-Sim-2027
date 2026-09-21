"""Generate a deterministic closed symmetric ellipsoidal plumbing-test hull."""

from pathlib import Path
import math


def generate(path: str|Path, *, length_m=2.0, beam_m=.4, draft_m=.15, rings=12, sectors=32) -> Path:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    a,b,c=length_m/2,beam_m/2,draft_m/2
    vertices=[]
    for i in range(1,rings):
        phi=-math.pi/2+math.pi*i/rings
        vertices.append([(a*math.sin(phi),b*math.cos(phi)*math.cos(2*math.pi*j/sectors),
                          c*math.cos(phi)*math.sin(2*math.pi*j/sectors)) for j in range(sectors)])
    south=(-a,0.,0.); north=(a,0.,0.); triangles=[]
    for j in range(sectors): triangles.append((south,vertices[0][(j+1)%sectors],vertices[0][j]))
    for i in range(len(vertices)-1):
        for j in range(sectors):
            k=(j+1)%sectors; triangles.extend(((vertices[i][j],vertices[i][k],vertices[i+1][k]),
                                                (vertices[i][j],vertices[i+1][k],vertices[i+1][j])))
    for j in range(sectors): triangles.append((vertices[-1][j],vertices[-1][(j+1)%sectors],north))
    lines=["solid cfd_smoke_hull"]
    for p,q,r in triangles:
        ux,uy,uz=(q[x]-p[x] for x in range(3)); vx,vy,vz=(r[x]-p[x] for x in range(3))
        nx,ny,nz=uy*vz-uz*vy,uz*vx-ux*vz,ux*vy-uy*vx; norm=math.sqrt(nx*nx+ny*ny+nz*nz)
        lines.append(f"  facet normal {nx/norm:.12g} {ny/norm:.12g} {nz/norm:.12g}")
        lines.append("    outer loop")
        lines.extend(f"      vertex {v[0]:.12g} {v[1]:.12g} {v[2]:.12g}" for v in (p,q,r))
        lines.extend(("    endloop","  endfacet"))
    lines.append("endsolid cfd_smoke_hull")
    path.write_text("\n".join(lines)+"\n")
    return path


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("output",type=Path)
    generate(parser.parse_args().output)
