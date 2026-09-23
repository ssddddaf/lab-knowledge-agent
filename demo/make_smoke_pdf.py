"""Generate a synthetic smoke-test PDF. It is not a research paper."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out=Path(__file__).with_name("synthetic_smoke.pdf")
fig,ax=plt.subplots(figsize=(8.3,11.7))
ax.axis("off")
ax.text(0.05,0.95,"SYNTHETIC DEMO REPORT",fontsize=22,transform=ax.transAxes)
ax.text(0.05,0.88,"Project: seismic_velocity_demo",fontsize=14,transform=ax.transAxes)
ax.text(0.05,0.83,"Run r2 uses first-arrival tomography and a 50 m grid.",fontsize=12,transform=ax.transAxes)
ax.text(0.05,0.78,"Run r2 is derived from r1; this is simulated data.",fontsize=12,transform=ax.transAxes)
ax.text(0.05,0.7,"Velocity model illustration",fontsize=14,transform=ax.transAxes)
ax.imshow([[2.0,2.3,2.7],[2.2,2.5,3.0],[2.4,2.8,3.2]],extent=[0.1,0.85,0.25,0.62],transform=ax.transAxes)
fig.savefig(out)
plt.close(fig)
print(out)
