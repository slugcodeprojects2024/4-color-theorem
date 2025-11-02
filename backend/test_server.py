"""Test the backend server."""
import requests
import numpy as np
from PIL import Image, ImageDraw
import io

def create_test_image():
    """Create a simple test coloring book image."""
    # Create white image
    img = Image.new('RGB', (400, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some regions
    # Rectangle
    draw.rectangle([50, 50, 150, 150], outline='black', width=3)
    # Circle
    draw.ellipse([200, 50, 300, 150], outline='black', width=3)
    # Triangle
    draw.polygon([(100, 200), (50, 300), (150, 300)], outline='black', width=3)
    # Another rectangle
    draw.rectangle([200, 200, 350, 350], outline='black', width=3)
    
    return img

def test_api():
    """Test the API endpoint."""
    # Create test image
    img = create_test_image()
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Send request
    files = {'file': ('test.png', img_bytes, 'image/png')}
    response = requests.post('http://localhost:8000/api/process', files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Stats: {data['stats']}")
        # You can save the result image if needed
        # base64_to_image(data['image'])
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_api()