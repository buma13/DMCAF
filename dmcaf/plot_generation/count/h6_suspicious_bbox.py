#!/usr/bin/env python3
"""
Hypothesis 6: Suspicious bounding boxes indicate overdetection and false positives.
Focus: confidence outliers (detections with confidence significantly lower than the image's mean).
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from .base_analyzer import BaseHypothesisAnalyzer, MODEL_COLORS, SUSPICION_ZSCORE_THRESHOLD

class Hypothesis6Analyzer(BaseHypothesisAnalyzer):
    """
    Analyzes suspicious bounding boxes to identify overdetection and false positives.
    """
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        super().__init__(metrics_db_path, analytics_db_path, output_dir)
        self.hypothesis_name = "H6_suspicious_bbox_detection"
        self.hypothesis_text = "Suspicious bounding boxes indicate overdetection and false positives"

    def run_analysis(self):
        """
        HYPOTHESIS 6: Suspicious bounding boxes indicate overdetection and false positives

        Method (confidence-focused): compute a continuous suspicion score that emphasizes
        low absolute confidence and low global percentile with a softer contribution from
        within-image negative z-score. No hard thresholds.
        """
        # Filter data with bbox information
        bbox_data = self.df[self.df['bbox_pixel_ratios'].notna()].copy()

        if len(bbox_data) == 0:
            return {'verified': False, 'reason': 'No data with bounding boxes available'}

        # Build global confidence distribution (for percentiles) without hard thresholds
        self.global_conf_sorted = self._build_global_conf_distribution(bbox_data)

        # Extract and analyze suspicious patterns (confidence-only focus)
        suspicious_bboxes = []
        for idx, row in bbox_data.iterrows():
            try:
                bbox_json = json.loads(row['bbox_pixel_ratios'])
                if bbox_json:
                    suspicious_bboxes.extend(self._analyze_image_bboxes(bbox_json, row))
            except (json.JSONDecodeError, TypeError):
                continue

        if len(suspicious_bboxes) == 0:
            return {'verified': False, 'reason': 'No suspicious bounding boxes found'}

        # Sort by suspicion score and get top 10 from different images
        suspicious_df = pd.DataFrame(suspicious_bboxes).sort_values('suspicion_score', ascending=False)

        # Select top suspicious bboxes ensuring diversity across images
        top_suspicious = self._select_diverse_suspicious_bboxes(suspicious_df, max_count=10)

        # Prepare collage entries for JSON summary (include paths and bbox coords)
        collage_entries = []
        for entry in top_suspicious:
            try:
                box = self._extract_box_xyxy(entry)
                collage_entries.append({
                    'image_path': entry.get('image_path'),
                    'yolo_image_path': str(self._convert_to_yolo_path(entry.get('image_path'))),
                    'bbox_xyxy': box,
                    'confidence': float(entry.get('confidence', 0.0)),
                    'conf_z': float(entry.get('conf_z', 0.0)),
                    'conf_mean_image': float(entry.get('image_conf_mean', 0.0)),
                    'confidence_gap': float(entry.get('confidence_gap', 0.0)),
                    'confidence_percentile': None if pd.isna(entry.get('confidence_percentile', np.nan)) else float(entry.get('confidence_percentile')),
                    'suspicion_score': int(entry.get('suspicion_score', 0)),
                    'reasons': entry.get('suspicion_reasons', []),
                    'model_short': entry.get('model_short'),
                    'expected_count': int(entry.get('expected_count', 0)),
                    'detected_count': int(entry.get('detected_count', 0)),
                    'count_accuracy': float(entry.get('count_accuracy', 0.0)),
                })
            except Exception:
                continue

        # Create visual analysis
        self._create_suspicious_bbox_collage(top_suspicious)

        # Statistical analysis
        correct_detections = suspicious_df[suspicious_df['count_accuracy'] == 1.0]
        overdetections = suspicious_df[suspicious_df['detected_count'] > suspicious_df['expected_count']]
        underdetections = suspicious_df[suspicious_df['detected_count'] < suspicious_df['expected_count']]

        # Compare suspicion scores across detection categories
        suspicion_stats = {
            'correct_mean': correct_detections['suspicion_score'].mean() if len(correct_detections) > 0 else 0,
            'over_mean': overdetections['suspicion_score'].mean() if len(overdetections) > 0 else 0,
            'under_mean': underdetections['suspicion_score'].mean() if len(underdetections) > 0 else 0,
            'total_suspicious': len(suspicious_df),
            'in_overdetections': len(overdetections),
            'in_correct': len(correct_detections)
        }

        # Create analysis plot and capture underlying data for JSON
        plot_data = {}
        if len(suspicious_df) > 0:
            # Suspicion by category (means only to keep JSON small)
            def _h6_category(row):
                if row['count_accuracy'] == 1.0:
                    return 'Correct'
                elif row['detected_count'] > row['expected_count']:
                    return 'Overdetection'
                else:
                    return 'Underdetection'
            tmp_df = suspicious_df.copy()
            tmp_df['category'] = tmp_df.apply(_h6_category, axis=1)
            by_cat_mean = tmp_df.groupby('category')['suspicion_score'].mean()
            plot_data['by_category_mean'] = {k: float(v) for k, v in by_cat_mean.to_dict().items()}

            # Note: Skip storing per-point scatter arrays to keep JSON compact

            # Reasons frequency
            all_reasons = []
            for reasons in suspicious_df['suspicion_reasons']:
                all_reasons.extend(reasons)
            reason_counts = pd.Series(all_reasons).value_counts()
            plot_data['reason_counts'] = reason_counts.to_dict()

            # Model averages
            model_suspicion = suspicious_df.groupby('model_short')['suspicion_score'].mean().sort_values(ascending=False)
            plot_data['model_avg_suspicion'] = {k: float(v) for k, v in model_suspicion.to_dict().items()}

            # Generate and save plots
            self._plot_suspicion_analysis(suspicious_df)

        # Determine if hypothesis is verified
        overdetection_enriched = (suspicion_stats['over_mean'] > suspicion_stats['correct_mean']) if suspicion_stats['correct_mean'] > 0 else False

        result = {
            'verified': overdetection_enriched and len(overdetections) > 0,
            'suspicion_stats': suspicion_stats,
            'top_suspicious_count': len(top_suspicious),
            'collage_entries': collage_entries,
            'plot_data': plot_data,
            'pattern_types': self._categorize_suspicion_patterns(suspicious_df),
            'conclusion': f"{'✅ VERIFIED' if overdetection_enriched else '❌ NOT VERIFIED'}: "
                         f"Overdetections {'show' if overdetection_enriched else 'do not show'} higher confidence-outlier scores"
        }

        self.hypothesis_results[self.hypothesis_name] = result
        return result

    def _build_global_conf_distribution(self, bbox_data_df: pd.DataFrame):
        """Collect all confidences across dataset and return a sorted numpy array for percentile lookup."""
        confs = []
        for _, row in bbox_data_df.iterrows():
            try:
                items = json.loads(row['bbox_pixel_ratios'])
                if isinstance(items, list):
                    confs.extend([it.get('confidence', 0.0) for it in items if isinstance(it, dict)])
            except Exception:
                continue
        if not confs:
            return np.array([], dtype=float)
        return np.array(sorted(confs), dtype=float)
    
    def _analyze_image_bboxes(self, bbox_json, row):
        """Analyze bounding boxes within a single image focusing on low-confidence outliers.

    Suspicion emphasizes low absolute confidence and global low percentile, with a softer
    contribution from within-image negative z-score. No hard gating by thresholds.
        """
        if len(bbox_json) <= 1:
            return []

        suspicious_bboxes = []

        # Extract confidence metrics for all bboxes in the image
        confidences = [bbox.get('confidence', 0) for bbox in bbox_json]

        # Calculate image-level statistics
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences) if len(confidences) > 1 else 0

        for i, bbox in enumerate(bbox_json):
            conf = bbox.get('confidence', 0.0)
            conf_z = (conf - conf_mean) / conf_std if conf_std > 0 else 0.0

            # Global percentile of confidence (0..1); lower = worse
            if getattr(self, 'global_conf_sorted', None) is not None and len(self.global_conf_sorted) > 0:
                rank = np.searchsorted(self.global_conf_sorted, conf, side='right')
                conf_pctl = rank / float(len(self.global_conf_sorted))
            else:
                conf_pctl = np.nan

            # Components in [0,1]
            comp_abs = (1.0 - conf)
            comp_abs = comp_abs * comp_abs  # square to accentuate lower values
            comp_pct = (1.0 - conf_pctl) if not np.isnan(conf_pctl) else 0.0
            comp_z = max(0.0, min(1.0, -conf_z / 3.0))  # ~1 at -3σ

            # Weighted continuous suspicion score mapped to 0..10
            score_float = 0.6 * comp_abs + 0.3 * comp_pct + 0.1 * comp_z
            suspicion_score = int(round(score_float * 10))
            # Ensure at least 1 to participate in ranking; avoid hard thresholding
            suspicion_score = max(1, suspicion_score)

            reasons = []
            if comp_abs > 0.25:
                reasons.append('Low Confidence (abs)')
            if comp_pct > 0.25:
                reasons.append('Low Percentile (global)')
            if conf_std > 0 and (-conf_z) > SUSPICION_ZSCORE_THRESHOLD:
                reasons.append('Within-image Low z')
            if not reasons:
                reasons.append('Mild Low Confidence')

            bbox_info = bbox.copy()
            bbox_info.update({
                'suspicion_score': suspicion_score,
                'suspicion_score_float': score_float,
                'suspicion_reasons': reasons,
                'image_path': row['image_path'],
                'model_short': row['model_short'],
                'expected_count': row['expected_count'],
                'detected_count': row['detected_count'],
                'count_accuracy': row['count_accuracy'],
                'conf_z': conf_z,
                'image_conf_mean': conf_mean,
                'confidence_gap': conf_mean - conf,
                'confidence_percentile': conf_pctl
            })
            suspicious_bboxes.append(bbox_info)

        return suspicious_bboxes
    
    def _select_diverse_suspicious_bboxes(self, suspicious_df, max_count=10):
        """Select top suspicious bboxes ensuring diversity across different images."""
        selected = []
        used_images = set()
        
        # First pass: one bbox per image, highest suspicion
        for _, row in suspicious_df.iterrows():
            if row['image_path'] not in used_images:
                selected.append(row.to_dict())
                used_images.add(row['image_path'])
                if len(selected) >= max_count:
                    break
        
        # Second pass: fill remaining slots with highest suspicion regardless of image
        if len(selected) < max_count:
            remaining_count = max_count - len(selected)
            existing_indices = [s['index'] for s in selected] if 'index' in suspicious_df.columns else []
            
            more_bboxes = suspicious_df[~suspicious_df.index.isin(existing_indices)].head(remaining_count)
            selected.extend(more_bboxes.to_dict('records'))
        
        return selected
    
    def _create_suspicious_bbox_collage(self, top_suspicious):
        """Create visual collage of suspicious bounding boxes with annotations."""
        if len(top_suspicious) == 0:
            return
        
        # Calculate collage dimensions
        cols = min(3, len(top_suspicious))
        rows = (len(top_suspicious) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, bbox_info in enumerate(top_suspicious):
            ax = axes[i]
            try:
                yolo_path = self._convert_to_yolo_path(bbox_info['image_path'])
                img = Image.open(yolo_path)
                
                # Draw bounding box
                draw = ImageDraw.Draw(img)
                box = self._extract_box_xyxy(bbox_info)
                if box is None:
                    raise KeyError('bbox coordinates not found')
                draw.rectangle(box, outline='red', width=3)
                
                ax.imshow(img)
                
                # Build a compact, scannable title with one score and key stats
                conf = bbox_info.get('confidence', 0.0)
                conf_z = bbox_info.get('conf_z', 0.0)
                gap = bbox_info.get('confidence_gap', None)
                mean_c = bbox_info.get('image_conf_mean', None)
                pctl = bbox_info.get('confidence_percentile', np.nan)
                score_f = bbox_info.get('suspicion_score_float', None)
                score_val = (score_f * 10.0) if score_f is not None else float(bbox_info.get('suspicion_score', 0))
                exp_c = bbox_info.get('expected_count', None)
                det_c = bbox_info.get('detected_count', None)

                line1 = f"Suspicion {score_val:.1f}/10"
                if exp_c is not None and det_c is not None:
                    line1 += f" | Exp/Det {int(exp_c)}/{int(det_c)}"

                # Conf line: Conf, z (signed), gap
                conf_bits = [f"Conf {conf:.2f}", f"z {conf_z:+.2f}"]
                if gap is not None:
                    conf_bits.append(f"gap {gap:.2f}")
                line2 = " | ".join(conf_bits)

                # Distribution line: percentile and image mean
                dist_bits = []
                if not np.isnan(pctl):
                    dist_bits.append(f"pctl {pctl*100:.0f}%")
                if mean_c is not None:
                    dist_bits.append(f"mean {mean_c:.2f}")
                line3 = " | ".join(dist_bits) if dist_bits else None

                # Optional: short reasons (kept minimal)
                reasons = bbox_info.get('suspicion_reasons', [])
                short_reasons = ", ".join(reasons[:2]) if reasons else None

                title = line1 + "\n" + line2
                if line3:
                    title += "\n" + line3
                if short_reasons:
                    title += "\n" + short_reasons
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            except FileNotFoundError:
                ax.text(0.5, 0.5, f"Image not found:\n{Path(bbox_info['image_path']).name}", ha='center', va='center')
                ax.axis('off')
            except KeyError:
                ax.text(0.5, 0.5, "Missing bbox coords", ha='center', va='center')
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(top_suspicious), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_6_suspicious_bbox_collage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _convert_to_yolo_path(self, image_path):
        """Convert generated image path to YOLO output path."""
        # Convert path: data_003/experiment_003/file.png -> runs/segment/predict/file.jpg
        image_path = Path(image_path)
        
        # Extract filename and change extension
        filename = image_path.stem + '.jpg'
        
        # Construct YOLO output path
        yolo_path = Path('yolo_outputs/count') / filename
        
        # Make it absolute based on current working directory
        return Path.cwd() / yolo_path

    def _extract_box_xyxy(self, bbox_info):
        """Return (x1, y1, x2, y2) from bbox_info supporting multiple key formats.
        Prefers 'bbox' as stored by Evaluator; falls back to 'box' or separate keys.
        Returns None if unavailable or invalid.
        """
        coords = None
        if isinstance(bbox_info, dict):
            if 'bbox' in bbox_info and isinstance(bbox_info['bbox'], (list, tuple)) and len(bbox_info['bbox']) == 4:
                coords = bbox_info['bbox']
            elif 'box' in bbox_info and isinstance(bbox_info['box'], (list, tuple)) and len(bbox_info['box']) == 4:
                coords = bbox_info['box']
            elif all(k in bbox_info for k in ('x1', 'y1', 'x2', 'y2')):
                coords = [bbox_info['x1'], bbox_info['y1'], bbox_info['x2'], bbox_info['y2']]
        if coords is None:
            return None
        try:
            x1, y1, x2, y2 = map(float, coords)
        except Exception:
            return None
        x1i, y1i, x2i, y2i = int(round(min(x1, x2))), int(round(min(y1, y2))), int(round(max(x1, x2))), int(round(max(y1, y2)))
        if x2i <= x1i or y2i <= y1i:
            return None
        return (x1i, y1i, x2i, y2i)
    
    def _plot_suspicion_analysis(self, suspicious_df):
        """Create statistical analysis plot focused on confidence outliers."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Suspicion score by detection category
        categories = []
        scores = []
        
        for _, row in suspicious_df.iterrows():
            if row['count_accuracy'] == 1.0:
                categories.append('Correct')
            elif row['detected_count'] > row['expected_count']:
                categories.append('Overdetection')
            else:
                categories.append('Underdetection')
            scores.append(row['suspicion_score'])
        
        suspicion_by_cat = pd.DataFrame({'category': categories, 'suspicion_score': scores})
        
        # Box plot of suspicion scores
        suspicion_by_cat.boxplot(column='suspicion_score', by='category', ax=ax1)
        ax1.set_title('Suspicion Scores by Detection Category')
        ax1.set_ylabel('Suspicion Score')
        plt.suptitle('')  # Remove default title
        
        # Plot 2: Confidence vs Outlier Magnitude (-z) colored by suspicion score
        outlier_mag = suspicious_df.get('conf_z', pd.Series([0]*len(suspicious_df)))
        if isinstance(outlier_mag, pd.Series):
            outlier_mag = outlier_mag.apply(lambda z: -z if z < 0 else 0)
        else:
            outlier_mag = pd.Series([0]*len(suspicious_df))
        scatter = ax2.scatter(suspicious_df['confidence'], outlier_mag,
                              c=suspicious_df['suspicion_score'], cmap='Reds', alpha=0.7)
        ax2.set_xlabel('YOLO Confidence')
        ax2.set_ylabel('Outlier Magnitude (-z)')
        ax2.set_title('Confidence vs Outlier Magnitude (Colored by Suspicion)')
        plt.colorbar(scatter, ax=ax2, label='Suspicion Score')
        
    # Plot 3: Suspicion reasons frequency (will primarily reflect confidence outliers)
        all_reasons = []
        for reasons in suspicious_df['suspicion_reasons']:
            all_reasons.extend(reasons)
        
        reason_counts = pd.Series(all_reasons).value_counts()
        reason_counts.plot(kind='bar', ax=ax3)
        ax3.set_title('Frequency of Suspicion Patterns')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Model-specific suspicion patterns
        model_suspicion = suspicious_df.groupby('model_short')['suspicion_score'].mean().sort_values(ascending=False)
        model_suspicion.plot(kind='bar', ax=ax4, color=[MODEL_COLORS.get(m, 'gray') for m in model_suspicion.index])
        ax4.set_title('Average Suspicion Score by Model')
        ax4.set_ylabel('Mean Suspicion Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_6_suspicion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _categorize_suspicion_patterns(self, suspicious_df):
        """Categorize types of suspicious patterns found."""
        pattern_counts = {}
        
        for reasons in suspicious_df['suspicion_reasons']:
            for reason in reasons:
                pattern_counts[reason] = pattern_counts.get(reason, 0) + 1
        
        return pattern_counts
