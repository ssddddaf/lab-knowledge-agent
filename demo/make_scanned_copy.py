"""Rasterize a source PDF to remove its text layer for retrieval testing."""
import argparse
from pathlib import Path

import fitz

parser=argparse.ArgumentParser()
parser.add_argument("source",type=Path)
parser.add_argument("output",type=Path)
parser.add_argument("--pages",type=int,default=3)
args=parser.parse_args()
source=fitz.open(args.source)
target=fitz.open()
for number in range(min(args.pages,len(source))):
    page=source[number]
    pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
    scan=target.new_page(width=page.rect.width,height=page.rect.height)
    scan.insert_image(scan.rect,stream=pix.tobytes("png"))
target.save(args.output)
target.close();source.close()
print(args.output)
