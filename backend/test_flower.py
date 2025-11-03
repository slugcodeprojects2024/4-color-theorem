# Create test_flower.py
import requests
import base64
from PIL import Image
import io

def test_flower():
    # Test with different styles
    styles = ['vibrant', 'pastel', 'earth', 'monochrome']
    
    for style in styles:
        with open('static/test_images/flower_coloring_book.png', 'rb') as f:
            files = {'file': f}
            data = {'style': style}
            response = requests.post('http://localhost:8000/api/process', 
                                   files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nStyle: {style}")
            print(f"Regions detected: {result['stats']['regions']}")
            print(f"Colors used: {result['stats']['colors_used']}")
            print(f"Graph edges: {result['stats']['graph_edges']}")
            
            # Save result
            img_data = base64.b64decode(result['image'].split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            img.save(f'static/test_images/flower_{style}.png')
            print(f"Saved: flower_{style}.png")

test_flower()