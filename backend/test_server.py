"""Test the backend server with debugging."""
import requests
import base64
from PIL import Image
import io
import json

def test_api_with_debug(image_path='static/test_images/grid_test.png'):
    """Test the API endpoint with debugging."""
    
    # Send request
    with open(image_path, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        response = requests.post('http://localhost:8000/api/process', files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Stats: {data['stats']}")
        
        # Save the result
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        output_name = image_path.replace('.png', '_colored.png')
        img.save(output_name)
        print(f"Saved result to: {output_name}")
        
        return data
    else:
        print(f"Error: {response.text}")
        return None

# Test different images
test_images = [
    'static/test_images/grid_test.png',
    'static/test_images/separate_boxes.png',
    'static/test_images/simple_coloring_page.png'
]

if __name__ == "__main__":
    for img_path in test_images:
        print(f"\n--- Testing {img_path} ---")
        try:
            test_api_with_debug(img_path)
        except FileNotFoundError:
            print(f"File not found: {img_path}")