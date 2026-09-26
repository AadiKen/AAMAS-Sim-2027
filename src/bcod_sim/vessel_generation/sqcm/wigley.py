"""Published Wigley double-body hull fixture for experimental SQCM gates."""
from __future__ import annotations

import numpy as np

from .source import SourcePanels


def wigley_half_width(x: np.ndarray, z: np.ndarray, *, length: float = 2.5,
                      beam: float = .25, draft: float = .156) -> np.ndarray:
    return beam/2 * (1-(2*x/length)**2) * (1-(z/draft)**2)


def wigley_source_panels(*, longitudinal: int = 30, vertical: int = 5,
                         length: float = 2.5, beam: float = .25,
                         draft: float = .156) -> tuple[SourcePanels, np.ndarray]:
    """600 panels for 30×5; mask identifies physical wetted half."""
    if longitudinal < 3 or vertical < 2:
        raise ValueError("Wigley grid too coarse")
    x = np.linspace(-length/2, length/2, longitudinal+1)
    depth = np.linspace(0., draft, vertical+1)
    quads, physical = [], []
    for mirror in (1., -1.):
        for side in (-1., 1.):
            for i in range(longitudinal):
                for j in range(vertical):
                    indices = ((i,j),(i+1,j),(i+1,j+1),(i,j+1))
                    q = np.zeros((4,3))
                    for k,(ii,jj) in enumerate(indices):
                        q[k]=[x[ii],side*wigley_half_width(x[ii],depth[jj],length=length,beam=beam,draft=draft),mirror*depth[jj]]
                    cx,cz=q[:,0].mean(),q[:,2].mean()
                    hx=-4*beam*cx/length**2*(1-(cz/draft)**2)
                    hz=-beam*cz/draft**2*(1-(2*cx/length)**2)
                    target=np.array([-hx,side,-hz])
                    if np.dot(np.cross(q[1]-q[0],q[2]-q[0]),target)<0:
                        q=q[::-1]
                    quads.append(q)
                    physical.append(mirror>0)
    return SourcePanels.from_corners(np.asarray(quads)),np.asarray(physical)
