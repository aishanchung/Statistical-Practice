import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.family':['Microsoft JhengHei','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','figure.dpi':140,'font.size':10})
COLORS=['#087f8c','#af5448','#16344a','#a38f55','#637b8c']
def save(fig,out,name):
    dest=Path(out)/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(dest/(name+'.svg'),bbox_inches='tight');plt.close(fig)
