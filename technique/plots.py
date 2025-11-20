"""
Visualization and plotting for BDKT experiments
Publication-quality figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BDKTPlotter:
    """Generate publication-quality plots for BDKT"""
    
    def __init__(self, figsize_base: Tuple[float, float] = (12, 8)):
        self.figsize_base = figsize_base
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_metrics_summary(
        self,
        aggregated_metrics: Dict[str, Tuple[float, float]],
        output_path: str = "bdkt_experiment_plots.png"
    ):
        """
        Plot metrics summary with error bars
        (1) Figure showing mean ± std for all metrics
        """
        logger.info(f"Generating metrics summary plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(aggregated_metrics.keys())
        means = [aggregated_metrics[m][0] for m in metric_names]
        stds = [aggregated_metrics[m][1] for m in metric_names]
        
        # Normalize to [0, 1] for visualization
        x_pos = np.arange(len(metric_names))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('BDKT Performance Metrics (Mean ± Std)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in metric_names], fontsize=11)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics plot to {output_path}")
        plt.close()
    
    def plot_mastery_uncertainty(
        self,
        student_idx: int,
        skill_mastery: np.ndarray,
        skill_uncertainty: np.ndarray,
        responses: np.ndarray,
        output_path: str = "bdkt_mastery_uncertainty.png"
    ):
        """
        Plot skill mastery + uncertainty (±1 std) for example learner
        (2) Curve showing mastery over time with confidence bands
        """
        logger.info(f"Generating mastery/uncertainty plot for student {student_idx}...")
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Select top 6 skills by variance
        skill_vars = np.var(skill_mastery, axis=0)
        top_skills = np.argsort(skill_vars)[-6:]
        
        for plot_idx, skill_id in enumerate(top_skills):
            ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            
            mastery = skill_mastery[:, skill_id]
            uncertainty = skill_uncertainty[:, skill_id]
            timesteps = np.arange(len(mastery))
            
            # Plot mastery with confidence band
            ax.plot(timesteps, mastery, 'b-', linewidth=2.5, label='Mastery', zorder=3)
            ax.fill_between(timesteps, mastery - uncertainty, mastery + uncertainty,
                           alpha=0.3, color='blue', label='±1 Std', zorder=2)
            
            # Mark correct/incorrect responses
            correct = responses == 1
            incorrect = responses == 0
            ax.scatter(timesteps[correct], mastery[correct], color='green', s=50, 
                      marker='o', label='Correct', zorder=4, alpha=0.7)
            ax.scatter(timesteps[incorrect], mastery[incorrect], color='red', s=50,
                      marker='x', label='Incorrect', zorder=4, alpha=0.7)
            
            ax.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax.set_ylabel('Mastery', fontsize=10, fontweight='bold')
            ax.set_title(f'Skill {skill_id}', fontsize=11, fontweight='bold')
            ax.set_ylim([-0.1, 1.1])
            ax.grid(alpha=0.3)
            
            if plot_idx == 0:
                ax.legend(loc='best', fontsize=9)
        
        # Add overall title
        fig.suptitle(f'Student {student_idx}: Skill Mastery & Uncertainty Over Time',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved mastery/uncertainty plot to {output_path}")
        plt.close()
    
    def plot_uncertainty_after_gaps(
        self,
        skill_mastery: np.ndarray,
        skill_uncertainty: np.ndarray,
        time_gaps: np.ndarray,
        gap_threshold: float = 3.0,
        output_path: str = "bdkt_uncertainty_gaps.png"
    ):
        """
        Plot showing increased uncertainty after long gaps
        (3) Curve showing uncertainty increase after gaps
        """
        logger.info(f"Generating uncertainty after gaps plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Identify long gaps
        long_gap_mask = time_gaps > gap_threshold
        short_gap_mask = time_gaps <= gap_threshold
        
        # Average uncertainty by gap size
        gap_bins = np.linspace(0, time_gaps.max(), 20)
        uncertainty_mean = np.zeros(len(gap_bins) - 1)
        uncertainty_std = np.zeros(len(gap_bins) - 1)
        gap_centers = np.zeros(len(gap_bins) - 1)
        
        for i in range(len(gap_bins) - 1):
            mask = (time_gaps >= gap_bins[i]) & (time_gaps < gap_bins[i+1])
            if np.sum(mask) > 0:
                uncertainty_mean[i] = np.mean(skill_uncertainty[mask])
                uncertainty_std[i] = np.std(skill_uncertainty[mask])
                gap_centers[i] = (gap_bins[i] + gap_bins[i+1]) / 2
        
        # Plot 1: Uncertainty vs gap size
        ax1.errorbar(gap_centers, uncertainty_mean, yerr=uncertainty_std,
                    fmt='o-', linewidth=2.5, markersize=8, capsize=5,
                    color='#E63946', ecolor='#A4161A', elinewidth=2)
        ax1.axvline(gap_threshold, color='gray', linestyle='--', linewidth=2, label=f'Gap threshold={gap_threshold}')
        ax1.set_xlabel('Time Gap (log scale)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Mean Skill Uncertainty', fontsize=11, fontweight='bold')
        ax1.set_title('Uncertainty Increases After Long Gaps', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: Distribution comparison
        unc_long = skill_uncertainty[long_gap_mask].flatten()
        unc_short = skill_uncertainty[short_gap_mask].flatten()
        
        ax2.hist(unc_short, bins=30, alpha=0.6, label=f'Short gaps (≤{gap_threshold})',
                color='#06A77D', edgecolor='black')
        ax2.hist(unc_long, bins=30, alpha=0.6, label=f'Long gaps (>{gap_threshold})',
                color='#D62828', edgecolor='black')
        ax2.set_xlabel('Skill Uncertainty', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Uncertainty Distribution by Gap Type', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved uncertainty after gaps plot to {output_path}")
        plt.close()


if __name__ == "__main__":
    # Test plots
    plotter = BDKTPlotter()
    
    # Test metrics plot
    metrics = {
        'auc': (0.85, 0.03),
        'accuracy': (0.78, 0.04),
        'rmse': (0.35, 0.02),
        'ece': (0.08, 0.02),
    }
    plotter.plot_metrics_summary(metrics, "test_metrics.png")
    
    # Test mastery plot
    seq_len = 100
    num_skills = 30
    skill_mastery = np.random.uniform(0.3, 0.9, (seq_len, num_skills))
    skill_uncertainty = np.random.uniform(0.05, 0.2, (seq_len, num_skills))
    responses = np.random.randint(0, 2, seq_len)
    plotter.plot_mastery_uncertainty(0, skill_mastery, skill_uncertainty, responses, "test_mastery.png")
    
    # Test gaps plot
    time_gaps = np.random.exponential(2, (seq_len, num_skills))
    plotter.plot_uncertainty_after_gaps(skill_mastery, skill_uncertainty, time_gaps.mean(axis=1), "test_gaps.png")
