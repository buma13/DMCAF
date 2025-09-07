#!/usr/bin/env python3
"""
Base class for hypothesis-driven analysis.
"""
import sqlite3
import pandas as pd
from pathlib import Path
import json
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Configure plotting for clarity
plt.style.use('default')
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (14, 10),
    'font.size': 12
})

# Constants for DMCAF framework
IGNORED_OBJECTS = {"brocolli"}  # Known problematic objects
YOLO_OBJECTS = {39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl"}
MODEL_COLORS = {
    'SD-v1-5': '#1f77b4',
    'SD-2-1': '#ff7f0e', 
    'SD-3-medium': '#2ca02c',
    'SD-3.5-medium': '#d62728'
}

# Critical thresholds based on DMCAF experience
CRITICAL_ACCURACY_THRESHOLD = 0.3  # Below this = critical failure
HIGH_PIXEL_RATIO_THRESHOLD = 0.8   # Above this = good segmentation
HIGH_CONFIDENCE_THRESHOLD = 0.8    # Above this = confident detection

# Suspicious bbox detection thresholds
PHANTOM_CONF_THRESHOLD = 0.7       # High confidence for phantom detection
PHANTOM_PIXEL_THRESHOLD = 0.3      # Low pixel ratio for phantoms
PHANTOM_SIZE_THRESHOLD = 5          # Small bbox area percentage for phantoms
CASCADE_QUALITY_DROP = 0.3          # Quality drop threshold for overgeneration cascade
SCALE_VARIANCE_THRESHOLD = 0.5      # High variance in object sizes
SUSPICION_ZSCORE_THRESHOLD = 2.0    # Z-score threshold for outlier detection

class BaseHypothesisAnalyzer:
    """Base class for hypothesis-driven analyzer for count adherence in diffusion models."""
    
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        self.metrics_db_path = metrics_db_path
        self.analytics_db_path = analytics_db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.df = None
        self.hypothesis_results = {}
        self.load_error = None
        
    def load_data(self):
        """Load and preprocess data from SQLite database with schema checks."""
        try:
            conn = sqlite3.connect(self.metrics_db_path)

            # Discover available columns in image_evaluations
            pragma = pd.read_sql_query("PRAGMA table_info('image_evaluations')", conn)
            if pragma.empty:
                self.load_error = "Table 'image_evaluations' not found in metrics DB"
                self.df = None
                conn.close()
                return
            available_cols = set(pragma['name'].tolist())

            # Minimal required columns to run count analyses end-to-end
            required_cols = {
                'model_name', 'variant', 'target_object', 'expected_count', 'detected_count',
                'count_accuracy', 'image_path', 'bbox_pixel_ratios'
            }
            if not required_cols.issubset(available_cols):
                missing = sorted(required_cols - available_cols)
                self.load_error = f"Missing required columns for count analysis: {', '.join(missing)}"
                self.df = None
                conn.close()
                return

            # Columns we prefer to fetch if present
            optional_cols = [
                'pixel_ratio', 'coverage_percentage', 'target_pixels', 'total_pixels',
                'guidance_scale', 'num_inference_steps', 'condition_type',
                'segmented_pixels_in_bbox', 'total_bbox_pixels',
                'min_bbox_pixel_ratio', 'max_bbox_pixel_ratio', 'std_bbox_pixel_ratio'
            ]

            select_cols = [
                'model_name', 'variant', 'target_object', 'expected_count', 'detected_count',
                'count_accuracy', 'image_path', 'bbox_pixel_ratios'
            ] + [c for c in optional_cols if c in available_cols]

            query = (
                "SELECT " + ", ".join(select_cols) +
                " FROM image_evaluations WHERE expected_count IS NOT NULL AND count_accuracy IS NOT NULL"
            )

            self.df = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            self.load_error = f"Failed to load metrics: {e}"
            self.df = None
            try:
                conn.close()
            except Exception:
                pass
        
        # Clean and prepare data
        if self.df is None:
            return
        self.df['model_short'] = (
            self.df['model_name']
            .str.split('/').str[-1]
            .str.replace('stable-diffusion-', 'SD-')
            .str.replace('-diffusers', '')
        )
        
        # Filter invalid data
        self.df = self.df[
            (self.df['count_accuracy'] >= 0) &
            (self.df['count_accuracy'] <= 1) &
            (~self.df['target_object'].isin(IGNORED_OBJECTS))
        ].copy()
        
        # Extract confidence data from bbox_pixel_ratios JSON
        if 'bbox_pixel_ratios' in self.df.columns:
            self.df['has_high_confidence_objects'] = self.df['bbox_pixel_ratios'].apply(self._extract_confidence_info)
        
        # Check for potential sorting issues in database
        self._check_data_distribution()
        
    def _check_data_distribution(self):
        """Check if data is sorted by model (potential issue for sampling)."""
        # Check if data appears to be sorted by model
        model_positions = []
        for model in self.df['model_short'].unique():
            model_indices = self.df[self.df['model_short'] == model].index
            model_positions.append({
                'model': model,
                'first_index': model_indices.min(),
                'last_index': model_indices.max(),
                'count': len(model_indices)
            })
        
        # Sort by first appearance
        model_positions.sort(key=lambda x: x['first_index'])
        
        # Check if models appear in blocks (indicating sorted data)
        is_sorted_by_model = True
        for i in range(len(model_positions) - 1):
            if model_positions[i]['last_index'] >= model_positions[i+1]['first_index']:
                is_sorted_by_model = False
                break
                
        if is_sorted_by_model:
            # Data is sorted, consider shuffling for robust sampling
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    def _get_balanced_sample(self, df, target_variants, samples_per_combination=50):
        """Get a balanced sample across models and variants."""
        balanced_data = []
        
        for model in df['model_short'].unique():
            for variant in target_variants:
                subset = df[(df['model_short'] == model) & (df['variant'] == variant)]
                if len(subset) > samples_per_combination:
                    balanced_data.append(subset.sample(samples_per_combination, random_state=42))
                else:
                    balanced_data.append(subset)
        
        return pd.concat(balanced_data, ignore_index=True) if balanced_data else pd.DataFrame()
        
    def _extract_confidence_info(self, bbox_json_str):
        """Extract confidence information from bbox JSON data."""
        if pd.isna(bbox_json_str):
            return False
        try:
            bbox_data = json.loads(bbox_json_str)
            if isinstance(bbox_data, list) and len(bbox_data) > 0:
                return any(item.get('confidence', 0) > HIGH_CONFIDENCE_THRESHOLD for item in bbox_data)
            return False
        except (json.JSONDecodeError, TypeError):
            return False

    def run_analysis(self):
        raise NotImplementedError("Subclasses must implement the 'run_analysis' method.")
