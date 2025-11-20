"""
Evaluation metrics for BDKT
AUC, Accuracy, RMSE, ECE with k-fold stratified validation
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BDKTEvaluator:
    """Compute evaluation metrics for BDKT"""
    
    @staticmethod
    def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC-ROC"""
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred)
    
    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Compute accuracy"""
        y_pred_binary = (y_pred >= threshold).astype(int)
        return accuracy_score(y_true, y_pred_binary)
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        Measures how well predicted probabilities match actual frequencies
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(y_true)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
            prop_in_bin = np.sum(in_bin) / total_samples
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_pred[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {
            'auc': BDKTEvaluator.compute_auc(y_true, y_pred),
            'accuracy': BDKTEvaluator.compute_accuracy(y_true, y_pred),
            'rmse': BDKTEvaluator.compute_rmse(y_true, y_pred),
            'ece': BDKTEvaluator.compute_ece(y_true, y_pred),
        }
        return metrics


class StratifiedKFoldValidator:
    """Stratified k-fold validation by learner"""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def split_by_student(
        self,
        sequences_x: List[np.ndarray],
        sequences_y: List[np.ndarray],
        student_ids: List[int],
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create stratified k-fold splits by student
        Ensures each fold has balanced student distribution
        """
        student_ids_array = np.array(student_ids)
        unique_students = np.unique(student_ids_array)
        
        # Create binary labels for stratification (based on average performance)
        student_labels = np.zeros(len(unique_students), dtype=int)
        for i, sid in enumerate(unique_students):
            mask = student_ids_array == sid
            avg_performance = np.mean([np.mean(y) for j, y in enumerate(sequences_y) if mask[j]])
            student_labels[i] = 1 if avg_performance > 0.5 else 0
        
        folds = []
        for train_idx, test_idx in self.skf.split(unique_students, student_labels):
            train_students = set(unique_students[train_idx])
            test_students = set(unique_students[test_idx])
            
            train_seq_idx = [i for i, sid in enumerate(student_ids_array) if sid in train_students]
            test_seq_idx = [i for i, sid in enumerate(student_ids_array) if sid in test_students]
            
            folds.append((train_seq_idx, test_seq_idx))
        
        logger.info(f"Created {len(folds)} stratified folds")
        return folds


class MetricsAggregator:
    """Aggregate metrics across folds"""
    
    @staticmethod
    def aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """
        Compute mean ± std across folds
        
        Returns:
            Dict with metric names as keys and (mean, std) tuples as values
        """
        metric_names = fold_metrics[0].keys()
        aggregated = {}
        
        for metric_name in metric_names:
            values = np.array([m[metric_name] for m in fold_metrics if not np.isnan(m[metric_name])])
            if len(values) > 0:
                aggregated[metric_name] = (np.mean(values), np.std(values))
            else:
                aggregated[metric_name] = (np.nan, np.nan)
        
        return aggregated
    
    @staticmethod
    def print_results(aggregated_metrics: Dict[str, Tuple[float, float]]):
        """Pretty print aggregated metrics"""
        logger.info("\n" + "="*50)
        logger.info("BDKT Evaluation Results (Mean ± Std)")
        logger.info("="*50)
        
        for metric_name, (mean, std) in aggregated_metrics.items():
            if not np.isnan(mean):
                logger.info(f"{metric_name.upper():12s}: {mean:.4f} ± {std:.4f}")
        
        logger.info("="*50 + "\n")


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.95, 0.85])
    
    metrics = BDKTEvaluator.evaluate_predictions(y_true, y_pred)
    print("Metrics:", metrics)
    
    # Test k-fold
    sequences_x = [np.random.randn(100, 30) for _ in range(20)]
    sequences_y = [np.random.randint(0, 2, 100) for _ in range(20)]
    student_ids = [i // 5 for i in range(20)]  # 4 students, 5 sequences each
    
    validator = StratifiedKFoldValidator(n_splits=5)
    folds = validator.split_by_student(sequences_x, sequences_y, student_ids)
    print(f"Folds: {len(folds)}")
