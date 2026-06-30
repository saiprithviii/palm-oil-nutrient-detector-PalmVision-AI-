import cv2
import numpy as np
from fpdf import FPDF
import base64

def get_base64_bin(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def generate_pdf_report(condition, confidence, details):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(46, 204, 113)
    pdf.rect(0, 0, 210, 40, 'F')
    
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(200, 20, txt="PALM LEAF DIAGNOSTIC REPORT", ln=True, align='C')
    
    # Content
    pdf.set_text_color(0, 0, 0)
    pdf.ln(30)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Analysis Result: {condition}", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"AI Confidence Score: {confidence:.2f}%", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Clinical Recommendations", ln=True)
    
    pdf.set_font("Arial", '', 11)
    pdf.ln(5)
    # Clean strings for PDF
    symp = details['symptoms'].encode('latin-1', 'ignore').decode('latin-1')
    prot = details['fertilizer'].encode('latin-1', 'ignore').decode('latin-1')
    
    pdf.multi_cell(0, 10, txt=f"Observed Symptoms: {symp}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Treatment Protocol: {prot}")
    
    return pdf.output(dest='S').encode('latin-1')