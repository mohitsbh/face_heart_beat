from fpdf import FPDF
import tempfile
import time

def generate_pdf_report(bpm, blood_group):
    try:
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Heartbeat & Blood Group Report", ln=True, align='C')

        pdf.set_draw_color(0, 0, 0)
        pdf.line(10, 25, 200, 25)
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, txt=f"Estimated BPM: {bpm}", ln=True)
        pdf.cell(200, 10, txt=f"Blood Group: {blood_group}", ln=True)

        status = "Normal" if 60 <= bpm <= 100 else "Check-up Suggested"
        pdf.cell(200, 10, txt=f"Status: {status}", ln=True)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Arial", '', 10)
        pdf.cell(200, 10, txt=f"Report generated on: {timestamp}", ln=True, align='R')

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None
