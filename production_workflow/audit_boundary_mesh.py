"""Read-only topology/BC audit of copied production mesh; no numerical solves."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parent
config = json.loads((root.parent / 'amundsen_production_config.json').read_text())
mesh = root / 'amundsen.msh'
lines = mesh.read_text().splitlines()
i = lines.index('$Nodes'); n = int(lines[i+1])
nodes = {int(r[0]): [float(r[1]), float(r[2])] for r in
         (s.split() for s in lines[i+2:i+2+n])}
i = lines.index('$Elements'); n = int(lines[i+1])
boundaries = defaultdict(list); triangles = []
for line in lines[i+2:i+2+n]:
    v = list(map(int, line.split())); typ, nt = v[1:3]
    tags, vertices = v[3:3+nt], v[3+nt:]
    if typ == 1:
        boundaries[tags[0]].append(tuple(vertices))
    elif typ == 2:
        triangles.append(vertices)
    else:
        raise ValueError(f'Unexpected element type {typ}')
edges = Counter(tuple(sorted((t[j], t[(j+1)%3]))) for t in triangles for j in range(3))
exterior = {e for e,c in edges.items() if c == 1}
tagged = Counter(tuple(sorted(e)) for es in boundaries.values() for e in es)
dirichlet = set(config['domain']['dirichlet_ids'])
front = set(boundaries) - dirichlet - set(config['domain']['side_ids'])
checks = {
    'mesh_hash_matches_production': hashlib.sha256(mesh.read_bytes()).hexdigest() == config['inputs']['mesh']['sha256'],
    'all_tags_1_through_11': set(boundaries) == set(range(1,12)),
    'every_exterior_edge_tagged_exactly_once': set(tagged) == exterior and all(c == 1 for c in tagged.values()),
    'triangles_manifold': all(c in (1,2) for c in edges.values()),
    'icepack_complement_is_2_and_4': front == {2,4},
    'no_side_wall_class': config['domain']['side_ids'] == [],
}
rows = []
for tag, es in sorted(boundaries.items()):
    xy = np.array([[nodes[a], nodes[b]] for a,b in es])
    deg = Counter(v for e in es for v in e)
    rows.append(dict(tag=tag, condition='terminus stress' if tag in front else 'prescribed velocity',
                     edges=len(es), length_km=float(np.linalg.norm(xy[:,1]-xy[:,0],axis=1).sum()/1000),
                     closed_loop=all(d == 2 for d in deg.values()),
                     bounds_m=[xy[:,:,0].min(),xy[:,:,1].min(),xy[:,:,0].max(),xy[:,:,1].max()]))
report = dict(checks=checks, passed=all(checks.values()), nodes=len(nodes), triangles=len(triangles),
              exterior_edges=len(exterior), boundaries=rows,
              mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
              geometry_sha256=hashlib.sha256((root/'amundsen.geo').read_bytes()).hexdigest())
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['passed'], report

fig, ax = plt.subplots(figsize=(8.2,8.5))
xy = np.array([nodes[k] for k in sorted(nodes)])/1000
idx = {k:i for i,k in enumerate(sorted(nodes))}
tri = np.array([[idx[k] for k in t] for t in triangles])
ax.triplot(xy[:,0], xy[:,1], tri, color='0.78', linewidth=0.24, zorder=1)
for tag, es in sorted(boundaries.items()):
    seg = np.array([[nodes[a],nodes[b]] for a,b in es])/1000
    color = '#d55e00' if tag in front else '#0072b2'
    ax.add_collection(LineCollection(seg,colors=color,linewidths=2.3,zorder=3))
    mid = seg[len(seg)//2].mean(axis=0)
    ax.annotate(str(tag),mid,xytext=(9,7),textcoords='offset points',fontsize=12,fontweight='bold',
                color=color,bbox=dict(boxstyle='round,pad=0.16',facecolor='white',edgecolor='none',alpha=0.94),zorder=5)
ax.set_aspect('equal'); ax.set_xlabel('EPSG:3031 x (km)',fontsize=13); ax.set_ylabel('EPSG:3031 y (km)',fontsize=13)
ax.tick_params(labelsize=11); ax.margins(0.06)
ax.set_title('Production mesh: boundary tags and conditions',fontsize=14,pad=14)
ax.legend(handles=[Line2D([0],[0],color='#0072b2',lw=2.5,label='Prescribed observed velocity'),
                   Line2D([0],[0],color='#d55e00',lw=2.5,label='Ocean-pressure terminus condition')],
          loc='upper center',bbox_to_anchor=(0.5,-0.10),fontsize=12,frameon=False)
fig.tight_layout(); fig.savefig(root/'boundary_tags.png',dpi=200,bbox_inches='tight'); plt.close(fig)
print(json.dumps(report,indent=2))
