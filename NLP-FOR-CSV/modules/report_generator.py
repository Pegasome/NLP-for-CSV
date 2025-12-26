from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image as PILImage

def generate_pdf_report(task_info, results, df):
    """Generate PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph(f"<b>Analysis Report: {task_info['task'].upper()}</b>", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Summary
    story.append(Paragraph(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns", styles['Normal']))
    story.append(Paragraph(f"Task: {task_info['task']} | Confidence: {task_info['confidence']}*", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Insights
    if 'insights' in results:
        for insight in results['insights']:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
    
    # Save figure if exists
    if 'figure' in results:
        img_buffer = BytesIO()
        plt.figure(figsize=(8, 6))
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        story.append(Image(img_buffer, 6*inch, 4*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
