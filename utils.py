import matplotlib.pyplot as plt
import numpy as np

def plot_results(lr_grace, hr_generated, inter=None, flag=False):
    # Ensure the tensors are on CPU and convert to numpy arrays
    lr_grace = lr_grace.detach().cpu().numpy()
    hr_generated = hr_generated.detach().cpu().numpy()
    tpb=np.load('tpb_h.npy')
    # Remove batch dimension
    lr_grace = np.squeeze(lr_grace)
    hr_generated = np.squeeze(hr_generated)
    #lr_grace[tpb==0]=np.nan
    hr_generated[tpb==0]=np.nan
    # Determine global min and max for consistent color scale
    vmin = min(lr_grace.min(), hr_generated.min(), inter.min() if inter is not None else float('inf'))
    vmax = max(lr_grace.max(), hr_generated.max(), inter.max() if inter is not None else float('-inf'))
    
    if inter is None:
        # Plot LR and HR images side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        im0 = axes[0].imshow(lr_grace, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Low Resolution GRACE')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(hr_generated, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('High Resolution GRACE')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    elif flag:
        inter[tpb==0]=np.nan
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axes[0].imshow(lr_grace, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title('0.5 degree GRACE')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(hr_generated, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title('Downscaled GRACE')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(inter, cmap='jet', vmin=vmin, vmax=vmax)
        axes[2].set_title('0.25 degree original GRACE')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()
    else:
        inter[tpb==0]=np.nan
        # Plot LR, HR, and original HR images side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axes[0].imshow(lr_grace, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title('Low Resolution GRACE')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(hr_generated, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title('High Resolution GRACE')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(inter, cmap='jet', vmin=vmin, vmax=vmax)
        axes[2].set_title('High Resolution original GRACE')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def evaluate_metrics(trues, preds):
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    return mse, mae, r2

from matplotlib.projections import PolarAxes
from matplotlib.projections.polar import PolarTransform
from matplotlib.ticker import MultipleLocator
from matplotlib.spines import Spine
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot

class TaylorDiagram(object):
    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5)):
        self.refstd = refstd
        self.fig = fig or plt.figure()
        self.srange = srange

        tr = PolarTransform()
        rlocs = np.concatenate((np.arange(10) / 10., [0.95, 0.99, 1.0]))
        tlocs = np.arccos(rlocs)
        gl1 = FixedLocator(tlocs)  # Correlation
        tf1 = DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        self.grid_helper = GridHelperCurveLinear(
            tr, extremes=(0, np.pi/2, srange[0], srange[1]),
            grid_locator1=gl1, tick_formatter1=tf1
        )

        ax = FloatingSubplot(self.fig, rect, grid_helper=self.grid_helper)
        self.fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        self.ax.plot(np.arccos(1), refstd, 'k*', ls='', ms=10, label=label)

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        return l

    def add_grid(self, *args, **kwargs):
        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(self.srange[0], self.srange[1]),
                             np.linspace(0, np.pi/2))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

    @property
    def samplePoints(self):
        return self.ax.get_children()