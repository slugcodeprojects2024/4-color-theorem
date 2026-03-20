# Example Gallery Images

## How to Add Your Images

### Step 1: Prepare Your Images

For best alignment in the split view, your **before** and **after** images should:
- Have the **same dimensions** (width and height)
- Use the **same file format** (PNG, JPG, etc.)
- Be saved with descriptive names (e.g., `my-art-before.png` and `my-art-after.png`)

**Tip:** If your images have different sizes, resize them to match before adding them here.

### Step 2: Place Images in This Folder

Copy your image files directly into this folder:
```
frontend/public/examples/
```

For example:
- `frontend/public/examples/flower-before.png`
- `frontend/public/examples/flower-after.png`

### Step 3: Update the Component

Open `frontend/src/components/EducationalPage.js` and find the `examples` array (around line 7). Update it with your image filenames:

```javascript
const examples = [
  {
    title: 'Flower Coloring',
    beforeSrc: '/examples/flower-before.png',
    afterSrc: '/examples/flower-after.png',
    beforeAlt: 'Before: uncolored flower line art',
    afterAlt: 'After: automatically colored flower',
  },
  {
    title: 'Landscape Scene',
    beforeSrc: '/examples/landscape-before.png',
    afterSrc: '/examples/landscape-after.png',
    beforeAlt: 'Before: uncolored landscape',
    afterAlt: 'After: automatically colored landscape',
  },
  // Add more examples as needed...
];
```

### Step 4: Verify

1. Make sure your React dev server is running
2. Navigate to the Educational page
3. Click the "Example Gallery" tab
4. Your images should appear side-by-side in a split view

## Image Alignment Tips

- **Same dimensions = Perfect alignment**: If both images are the same size, they'll align perfectly
- **Different sizes**: The images will still display, but may not align perfectly. Consider resizing them to match
- **Aspect ratio**: Images maintain their aspect ratio, so if one is wider/taller, it will show more/less content

## File Naming Convention

Use clear, descriptive names:
- ✅ `flower-before.png` / `flower-after.png`
- ✅ `coloring-book-page-1-before.jpg` / `coloring-book-page-1-after.jpg`
- ❌ Avoid: `img1.png` / `img2.png` (not descriptive)
