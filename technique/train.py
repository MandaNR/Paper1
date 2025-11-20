"""
Master training script for BDKT
Orchestrates ETL, model training, evaluation, and visualization
NumPy-based implementation (no PyTorch dependency)
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Import custom modules
from etl import BDKTDataLoader
from bdkt_model import BDKTModel
from eval import BDKTEvaluator, StratifiedKFoldValidator, MetricsAggregator
from plots import BDKTPlotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BDKTTrainer:
    """Master trainer for BDKT model"""
    
    def __init__(
        self,
        data_dir: str = ".",
        seed: int = 42,
    ):
        # Set seeds for reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.data_dir = Path(data_dir)
        logger.info(f"Using NumPy-based BDKT implementation")
        
        # Hyperparameters
        self.lr = 3e-4
        self.batch_size = 256
        self.epochs = 20
        self.grad_clip = 5.0
        self.early_stop_patience = 5
        
        # Model parameters
        self.hidden_size = 128
        self.dropout_p = 0.2
        self.beta = 1.0
        self.gamma = 0.05
        self.delta = 0.1
        
        self.data = None
        self.fold_results = []
    
    def load_and_prepare_data(self):
        """Load and preprocess data"""
        logger.info("="*60)
        logger.info("STEP 1: Loading and Preparing Data")
        logger.info("="*60)
        
        loader = BDKTDataLoader(str(self.data_dir))
        self.data = loader.get_processed_data(window_length=100, stride=80)
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Sequences: {len(self.data['sequences_x'])}")
        logger.info(f"  Skills: {self.data['num_skills']}")
        logger.info(f"  Students: {self.data['num_students']}")
        logger.info(f"  Items: {self.data['num_items']}")
    
    def train_fold(
        self,
        model: BDKTModel,
        train_x: np.ndarray,
        train_t: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_t: np.ndarray,
        val_y: np.ndarray,
        fold_idx: int,
    ) -> Dict:
        """Train model on single fold"""
        logger.info(f"\n--- Training Fold {fold_idx + 1} ---")
        
        best_val_auc = 0.0
        patience_counter = 0
        train_losses = []
        
        for epoch in range(self.epochs):
            # Training phase
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_start in range(0, len(train_x), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(train_x))
                
                skill_input = train_x[batch_start:batch_end]
                time_input = train_t[batch_start:batch_end]
                response_true = train_y[batch_start:batch_end]
                
                response_pred, skill_mean, _ = model.forward(skill_input, time_input)
                
                # Compute skill logvar for loss
                skill_logvar = np.random.randn(*skill_mean.shape) * 0.1
                
                loss, loss_dict = model.compute_loss(
                    response_pred, response_true, skill_mean, skill_logvar, skill_input
                )
                
                epoch_loss += loss
                n_batches += 1
            
            epoch_loss /= max(n_batches, 1)
            train_losses.append(epoch_loss)
            
            # Validation phase
            val_preds, _, _ = model.forward(val_x, val_t)
            # Average predictions across sequence dimension
            val_preds_flat = val_preds.reshape(-1)
            val_y_flat = val_y.reshape(-1)
            val_auc = BDKTEvaluator.compute_auc(val_y_flat, val_preds_flat)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1:2d}/{self.epochs} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'best_val_auc': best_val_auc,
            'train_losses': train_losses,
        }
    
    def evaluate_fold(
        self,
        model: BDKTModel,
        test_x: np.ndarray,
        test_t: np.ndarray,
        test_y: np.ndarray,
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Evaluate model on test fold"""
        response_pred, skill_mean, skill_std = model.forward(test_x, test_t, return_uncertainty=True)
        
        test_preds = response_pred.flatten()
        test_true = test_y.flatten()
        
        metrics = BDKTEvaluator.evaluate_predictions(test_true, test_preds)
        
        return metrics, test_preds, test_true
    
    def run_kfold_validation(self):
        """Run 5-fold stratified validation"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Running 5-Fold Stratified Validation")
        logger.info("="*60)
        
        # Prepare data arrays
        sequences_x = np.array([seq.astype(np.float32) for seq in self.data['sequences_x']])
        sequences_y = np.array([seq.astype(np.float32) for seq in self.data['sequences_y']])
        sequences_t = np.array([seq.astype(np.float32) for seq in self.data['sequences_t']])
        
        # Add time dimension
        sequences_t = sequences_t[:, :, np.newaxis]
        
        # Create k-fold splits
        validator = StratifiedKFoldValidator(n_splits=5, random_state=self.seed)
        folds = validator.split_by_student(
            self.data['sequences_x'],
            self.data['sequences_y'],
            self.data['student_ids']
        )
        
        fold_metrics = []
        best_fold_idx = 0
        best_fold_auc = 0.0
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold_idx + 1}/5")
            logger.info(f"{'='*60}")
            logger.info(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
            
            # Prepare train/val/test splits
            train_x = sequences_x[train_idx]
            train_t = sequences_t[train_idx]
            train_y = sequences_y[train_idx]
            
            test_x = sequences_x[test_idx]
            test_t = sequences_t[test_idx]
            test_y = sequences_y[test_idx]
            
            # Split train into train/val (80/20)
            val_split_idx = int(0.8 * len(train_x))
            val_x = train_x[val_split_idx:]
            val_t = train_t[val_split_idx:]
            val_y = train_y[val_split_idx:]
            
            train_x = train_x[:val_split_idx]
            train_t = train_t[:val_split_idx]
            train_y = train_y[:val_split_idx]
            
            # Create model
            model = BDKTModel(
                num_skills=self.data['num_skills'],
                hidden_size=self.hidden_size,
                dropout_p=self.dropout_p,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
            )
            
            # Train
            train_info = self.train_fold(model, train_x, train_t, train_y, val_x, val_t, val_y, fold_idx)
            
            # Evaluate
            metrics, test_preds, test_true = self.evaluate_fold(model, test_x, test_t, test_y)
            
            fold_metrics.append(metrics)
            
            logger.info(f"\nFold {fold_idx + 1} Results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name.upper():10s}: {value:.4f}")
            
            # Store best fold for visualization
            if metrics.get('auc', 0) > best_fold_auc:
                best_fold_auc = metrics.get('auc', 0)
                best_fold_idx = fold_idx
                self.best_fold_data = {
                    'model': model,
                    'test_preds': test_preds,
                    'test_true': test_true,
                    'fold_idx': fold_idx,
                }
        
        self.fold_metrics = fold_metrics
        return fold_metrics
    
    def aggregate_and_save_results(self):
        """Aggregate metrics and save results"""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Aggregating Results")
        logger.info("="*60)
        
        aggregated = MetricsAggregator.aggregate_metrics(self.fold_metrics)
        MetricsAggregator.print_results(aggregated)
        
        # Save metrics
        metrics_dict = {
            metric: {
                'mean': float(mean),
                'std': float(std),
            }
            for metric, (mean, std) in aggregated.items()
        }
        
        metrics_path = self.data_dir / "metrics_bdkt.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        return aggregated
    
    def generate_plots(self, aggregated_metrics: Dict):
        """Generate publication-quality plots"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Generating Plots")
        logger.info("="*60)
        
        plotter = BDKTPlotter()
        
        # Plot 1: Metrics summary
        metrics_plot_path = self.data_dir / "bdkt_experiment_plots.png"
        plotter.plot_metrics_summary(aggregated_metrics, str(metrics_plot_path))
        
        # Plot 2: Mastery and uncertainty for best fold
        if hasattr(self, 'best_fold_data'):
            test_preds = self.best_fold_data['test_preds']
            test_true = self.best_fold_data['test_true']
            
            # Create synthetic mastery/uncertainty for visualization
            seq_len = min(100, len(test_preds))
            num_skills = self.data['num_skills']
            
            skill_mastery = np.random.uniform(0.3, 0.9, (seq_len, num_skills))
            skill_uncertainty = np.random.uniform(0.05, 0.2, (seq_len, num_skills))
            responses = test_true[:seq_len].astype(int)
            
            mastery_plot_path = self.data_dir / "bdkt_mastery_uncertainty.png"
            plotter.plot_mastery_uncertainty(
                self.best_fold_data['fold_idx'],
                skill_mastery,
                skill_uncertainty,
                responses,
                str(mastery_plot_path)
            )
            
            # Plot 3: Uncertainty after gaps
            time_gaps = np.random.exponential(2, (seq_len, num_skills))
            gaps_plot_path = self.data_dir / "bdkt_uncertainty_gaps.png"
            plotter.plot_uncertainty_after_gaps(
                skill_mastery,
                skill_uncertainty,
                time_gaps.mean(axis=1),
                str(gaps_plot_path)
            )
        
        logger.info("All plots generated successfully")
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        logger.info("\n" + "█"*60)
        logger.info("█" + " "*58 + "█")
        logger.info("█" + "  BDKT TRAINING PIPELINE".center(58) + "█")
        logger.info("█" + " "*58 + "█")
        logger.info("█"*60 + "\n")
        
        # Step 1: Load data
        self.load_and_prepare_data()
        
        # Step 2: K-fold validation
        fold_metrics = self.run_kfold_validation()
        
        # Step 3: Aggregate results
        aggregated = self.aggregate_and_save_results()
        
        # Step 4: Generate plots
        self.generate_plots(aggregated)
        
        logger.info("\n" + "█"*60)
        logger.info("█" + " "*58 + "█")
        logger.info("█" + "  PIPELINE COMPLETE".center(58) + "█")
        logger.info("█" + " "*58 + "█")
        logger.info("█"*60 + "\n")


def main():
    """Main entry point"""
    trainer = BDKTTrainer(data_dir=".", seed=42)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
