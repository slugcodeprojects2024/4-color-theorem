# Create a new file: create_better_test_image.py
from PIL import Image, ImageDraw
import os

os.makedirs('static/test_images', exist_ok=True)

# Create a simple coloring book image with clear separate regions
img = Image.new('RGB', (400, 400), 'white')
draw = ImageDraw.Draw(img)

# Draw thick black lines to create clear regions
line_width = 8

# Create a 2x2 grid
draw.line([(200, 0), (200, 400)], fill='black', width=line_width)
draw.line([(0, 200), (400, 200)], fill='black', width=line_width)

# Add a circle in the center that overlaps all quadrants
draw.ellipse([100, 100, 300, 300], outline='black', width=line_width)

# Save the test image
img.save('static/test_images/grid_test.png')
print("Test image created at: static/test_images/grid_test.png")

# Also create the simple one from before
img2 = Image.new('RGB', (400, 400), 'white')
draw2 = ImageDraw.Draw(img2)

# Draw separate shapes that don't touch
draw2.rectangle([50, 50, 150, 150], outline='black', width=5, fill=None)
draw2.rectangle([250, 50, 350, 150], outline='black', width=5, fill=None)
draw2.rectangle([50, 250, 150, 350], outline='black', width=5, fill=None)
draw2.rectangle([250, 250, 350, 350], outline='black', width=5, fill=None)

img2.save('static/test_images/separate_boxes.png')
print("Test image created at: static/test_images/separate_boxes.png")