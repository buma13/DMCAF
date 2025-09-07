#!/usr/bin/env python3
"""
Hypothesis 6: Suspicious bounding boxes indicate overdetection and false positives.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from .base_analyzer import BaseHypothesisAnalyzer, MODEL_COLORS, PHANTOM_CONF_THRESHOLD, PHANTOM_PIXEL_THRESHOLD, PHANTOM_SIZE_THRESHOLD, SUSPICION_ZSCORE_THRESHOLD

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
        
        Method: Identify suspicious bounding boxes with anomalous confidence, pixel ratio, 
        or size patterns compared to normal detections, then create visual collages for validation
        """
        # Filter data with bbox information
        bbox_data = self.df[self.df['bbox_pixel_ratios'].notna()].copy()
        
        if len(bbox_data) == 0:
            return {'verified': False, 'reason': 'No data with bounding boxes available'}
        
        # Extract and analyze suspicious patterns
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
        suspicious_df = pd.DataFrame(suspicious_bboxes)
        suspicious_df = suspicious_df.sort_values('suspicion_score', ascending=False)
        
        # Select top suspicious bboxes ensuring diversity across images
        top_suspicious = self._select_diverse_suspicious_bboxes(suspicious_df, max_count=10)
        
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
        
        # Create analysis plot
        if len(suspicious_df) > 0:
            self._plot_suspicion_analysis(suspicious_df)
        
        # Determine if hypothesis is verified
        overdetection_enriched = (suspicion_stats['over_mean'] > suspicion_stats['correct_mean']) if suspicion_stats['correct_mean'] > 0 else False
        
        result = {
            'verified': overdetection_enriched and len(overdetections) > 0,
            'suspicion_stats': suspicion_stats,
            'top_suspicious_count': len(top_suspicious),
            'pattern_types': self._categorize_suspicion_patterns(suspicious_df),
            'conclusion': f"{'✅ VERIFIED' if overdetection_enriched else '❌ NOT VERIFIED'}: "
                         f"Overdetections {'show' if overdetection_enriched else 'do not show'} higher suspicion scores"
        }
        
        self.hypothesis_results[self.hypothesis_name] = result
        return result
    
    def _analyze_image_bboxes(self, bbox_json, row):
        """Analyze bounding boxes within a single image for suspicious patterns."""
        if len(bbox_json) <= 1:
            return []
        
        suspicious_bboxes = []
        
        # Extract metrics for all bboxes in the image
        confidences = [bbox.get('confidence', 0) for bbox in bbox_json]
        pixel_ratios = [bbox.get('pixel_ratio', 0) for bbox in bbox_json]
        area_percentages = [bbox.get('bbox_area_percentage', 0) for bbox in bbox_json]
        
        # Calculate image-level statistics
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences) if len(confidences) > 1 else 0
        pixel_mean = np.mean(pixel_ratios)
        pixel_std = np.std(pixel_ratios) if len(pixel_ratios) > 1 else 0
        area_mean = np.mean(area_percentages)
        area_std = np.std(area_percentages) if len(area_percentages) > 1 else 0
        
        for i, bbox in enumerate(bbox_json):
            suspicion_score = 0
            reasons = []
            
            conf = bbox.get('confidence', 0)
            pr = bbox.get('pixel_ratio', 0)
            area = bbox.get('bbox_area_percentage', 0)
            
            # Pattern 1: Phantom object (high confidence, low pixel ratio)
            if conf > PHANTOM_CONF_THRESHOLD and pr < PHANTOM_PIXEL_THRESHOLD:
                suspicion_score += 3
                reasons.append('Phantom Object')
            
            # Pattern 2: Small, low-quality bbox
            if area < PHANTOM_SIZE_THRESHOLD and pr < PHANTOM_PIXEL_THRESHOLD:
                suspicion_score += 2
                reasons.append('Small & Low Quality')
            
            # Pattern 3: Statistical outlier (Z-score)
            conf_z = (conf - conf_mean) / conf_std if conf_std > 0 else 0
            pr_z = (pr - pixel_mean) / pixel_std if pixel_std > 0 else 0
            area_z = (area - area_mean) / area_std if area_std > 0 else 0
            
            if abs(conf_z) > SUSPICION_ZSCORE_THRESHOLD:
                suspicion_score += 1
                reasons.append('Confidence Outlier')
            if abs(pr_z) > SUSPICION_ZSCORE_THRESHOLD:
                suspicion_score += 1
                reasons.append('Pixel Ratio Outlier')
            if abs(area_z) > SUSPICION_ZSCORE_THRESHOLD:
                suspicion_score += 1
                reasons.append('Size Outlier')
            
            if suspicion_score > 0:
                bbox_info = bbox.copy()
                bbox_info.update({
                    'suspicion_score': suspicion_score,
                    'suspicion_reasons': reasons,
                    'image_path': row['image_path'],
                    'model_short': row['model_short'],
                    'expected_count': row['expected_count'],
                    'detected_count': row['detected_count'],
                    'count_accuracy': row['count_accuracy']
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
                
                title = f"Score: {bbox_info['suspicion_score']}\n"
                title += f"Reasons: {', '.join(bbox_info['suspicion_reasons'])}\n"
                title += f"Conf: {bbox_info['confidence']:.2f}, PR: {bbox_info['pixel_ratio']:.2f}, Area: {bbox_info['bbox_area_percentage']:.2f}%"
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
        """Create statistical analysis plot of suspicion patterns."""
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
        
        # Plot 2: Confidence vs Pixel Ratio colored by suspicion
        scatter = ax2.scatter(suspicious_df['confidence'], suspicious_df['pixel_ratio'], 
                            c=suspicious_df['suspicion_score'], cmap='Reds', alpha=0.7)
        ax2.set_xlabel('YOLO Confidence')
        ax2.set_ylabel('Pixel Ratio')
        ax2.set_title('Confidence vs Pixel Ratio (Colored by Suspicion)')
        plt.colorbar(scatter, ax=ax2, label='Suspicion Score')
        
        # Plot 3: Suspicion reasons frequency
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
