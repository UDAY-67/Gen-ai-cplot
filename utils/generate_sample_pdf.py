"""
Generate a sample PDF for testing GenAI Study Copilot
"""

def generate_minimal_pdf(output_path: str = "data/sample_study_guide.pdf"):
    # Standard minimal PDF structure with 3 pages of CS content
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >> endobj\n"
        # Page 1
        "3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 6 0 R >> >> /MediaBox [0 0 612 792] /Contents 7 0 R >> endobj\n"
        # Page 2
        "4 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 6 0 R >> >> /MediaBox [0 0 612 792] /Contents 8 0 R >> endobj\n"
        # Page 3
        "5 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 6 0 R >> >> /MediaBox [0 0 612 792] /Contents 9 0 R >> endobj\n"
        # Font
        "6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        # Page 1 Contents
        "7 0 obj << /Length 380 >> stream\n"
        "BT\n"
        "/F1 16 Tf\n"
        "50 720 Td\n"
        "(Chapter 1: Optimization and Gradient Descent) Tj\n"
        "/F1 11 Tf\n"
        "0 -30 Td\n"
        "(Gradient Descent is an iterative optimization algorithm used in Machine Learning) Tj\n"
        "0 -20 Td\n"
        "(to minimize the loss function J(theta). The update rule is theta := theta - alpha * grad(J).) Tj\n"
        "0 -20 Td\n"
        "(The learning rate alpha controls step size. If alpha is too large, it diverges.) Tj\n"
        "0 -20 Td\n"
        "(Stochastic Gradient Descent (SGD) updates weights using a single sample per step.) Tj\n"
        "ET\n"
        "endstream\n"
        "endobj\n"
        # Page 2 Contents
        "8 0 obj << /Length 380 >> stream\n"
        "BT\n"
        "/F1 16 Tf\n"
        "50 720 Td\n"
        "(Chapter 2: Support Vector Machines and Kernel Trick) Tj\n"
        "/F1 11 Tf\n"
        "0 -30 Td\n"
        "(Support Vector Machines (SVM) find the optimal hyperplane maximizing the geometric margin) Tj\n"
        "0 -20 Td\n"
        "(between two classes. The margin width is given by 2 / ||w||.) Tj\n"
        "0 -20 Td\n"
        "(The Kernel Trick maps input vectors into high-dimensional feature spaces using Mercer kernels.) Tj\n"
        "0 -20 Td\n"
        "(Popular kernels include Radial Basis Function (RBF), Linear, and Polynomial kernels.) Tj\n"
        "ET\n"
        "endstream\n"
        "endobj\n"
        # Page 3 Contents
        "9 0 obj << /Length 380 >> stream\n"
        "BT\n"
        "/F1 16 Tf\n"
        "50 720 Td\n"
        "(Chapter 3: Self-Attention in Transformer Architectures) Tj\n"
        "/F1 11 Tf\n"
        "0 -30 Td\n"
        "(The Scaled Dot-Product Attention equation is Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.) Tj\n"
        "0 -20 Td\n"
        "(Multi-Head Attention enables the model to jointly attend to information from different) Tj\n"
        "0 -20 Td\n"
        "(representation subspaces at different positions.) Tj\n"
        "0 -20 Td\n"
        "(Positional Encodings inject token order information into input embeddings.) Tj\n"
        "ET\n"
        "endstream\n"
        "endobj\n"
        "xref\n"
        "0 10\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000133 00000 n \n"
        "0000000257 00000 n \n"
        "0000000381 00000 n \n"
        "0000000505 00000 n \n"
        "0000000582 00000 n \n"
        "0000001015 00000 n \n"
        "0000001448 00000 n \n"
        "trailer << /Size 10 /Root 1 0 R >>\n"
        "startxref\n"
        "1881\n"
        "%%EOF\n"
    )
    with open(output_path, "wb") as f:
        f.write(pdf_content.encode("latin-1"))

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_study_guide.pdf"
    generate_minimal_pdf(path)
    print(f"Sample PDF created at {path}")
