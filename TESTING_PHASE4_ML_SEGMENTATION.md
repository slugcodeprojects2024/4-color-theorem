# Testing Guide: ML-Enhanced Segmentation (Phase 4.1)

## Overview
This guide helps you test the ML segmentation features that have been implemented. These features are part of Phase 4.1 (ML-Enhanced Region Detection), not Phase 3.

## What Was Implemented

1. **ML Segmentation Toggle** - Enable/disable ML-enhanced segmentation
2. **Segmentation Method Picker** - Choose between:
   - `auto` - Automatically selects best method
   - `slic` - SLIC superpixels (best for photos)
   - `edge` - Traditional edge detection (best for line art)
3. **Target Regions Slider** - Control how many regions to detect (default: 50)

## Test Checklist

### ✅ Basic Functionality Tests

#### Test 1: UI Components
- [ ] **Segmentation Settings Panel** appears in the controls section
- [ ] **Toggle checkbox** for "Use ML-Enhanced Segmentation" is visible
- [ ] When ML toggle is **OFF**, method picker and target regions are hidden
- [ ] When ML toggle is **ON**, method picker and target regions appear
- [ ] **Method dropdown** shows: Auto, SLIC, Edge
- [ ] **Target Regions slider** shows current value and allows adjustment (range: 20-200)

#### Test 2: Default Behavior
- [ ] ML segmentation is **disabled by default** (checkbox unchecked)
- [ ] Default method is `auto`
- [ ] Default target regions is `50`
- [ ] Without ML enabled, traditional edge detection is used

### ✅ ML Segmentation Tests

#### Test 3: Simple Line Art (ML OFF - Should Work)
- [ ] Upload a simple coloring book image (black lines, white background)
- [ ] Keep ML segmentation **OFF**
- [ ] Process the image
- [ ] **Expected**: Should work well, detecting distinct regions
- [ ] Check stats show reasonable number of regions

#### Test 4: Simple Line Art (ML ON - Edge Method)
- [ ] Upload the same simple coloring book image
- [ ] Enable ML segmentation
- [ ] Select method: **Edge**
- [ ] Process the image
- [ ] **Expected**: Should work similarly to traditional method
- [ ] Compare results with Test 3

#### Test 5: Complex Photo (ML ON - SLIC Method)
- [ ] Upload a complex photo (landscape, portrait, etc.)
- [ ] Enable ML segmentation
- [ ] Select method: **SLIC**
- [ ] Set target regions to **100**
- [ ] Process the image
- [ ] **Expected**: Should detect more regions than traditional method
- [ ] Check that output has multiple distinct colors (not mostly one color)
- [ ] Verify stats show appropriate number of regions

#### Test 6: Complex Photo (ML ON - Auto Method)
- [ ] Upload the same complex photo
- [ ] Enable ML segmentation
- [ ] Select method: **Auto**
- [ ] Process the image
- [ ] **Expected**: Should auto-select SLIC for photos
- [ ] Compare results with Test 5

#### Test 7: Target Regions Adjustment
- [ ] Upload a complex photo
- [ ] Enable ML segmentation, method: **SLIC**
- [ ] Test with **target regions = 30** (fewer regions)
- [ ] Process and note the number of regions detected
- [ ] Test with **target regions = 100** (more regions)
- [ ] Process and compare
- [ ] **Expected**: Higher target regions should produce more distinct regions

### ✅ Edge Cases & Error Handling

#### Test 8: Very Large Image
- [ ] Upload a large image (3000x3000px or larger)
- [ ] Enable ML segmentation
- [ ] Process the image
- [ ] **Expected**: Should handle without timeout or memory errors

#### Test 9: Very Small Image
- [ ] Upload a small image (200x200px)
- [ ] Enable ML segmentation
- [ ] Process the image
- [ ] **Expected**: Should still work, may use edge method automatically

#### Test 10: Image with Gradients (Sunset/Misty Landscape)
- [ ] Upload an image with smooth gradients (sunset, misty landscape)
- [ ] Enable ML segmentation, method: **SLIC**
- [ ] Set target regions to **150**
- [ ] Process the image
- [ ] **Expected**: Should produce multiple distinct regions (not mostly one color)
- [ ] If it fails, try with traditional method (ML OFF) - should work better

#### Test 11: Mixed Content Image
- [ ] Upload an image with both line art and photo elements
- [ ] Test with ML **OFF** and ML **ON** (auto method)
- [ ] Compare results
- [ ] **Expected**: Auto method should select appropriate technique

### ✅ Integration Tests

#### Test 12: With Line Art Converter
- [ ] Upload a photo
- [ ] Enable **Line Art Converter**
- [ ] Enable **ML Segmentation** (method: SLIC)
- [ ] Process the image
- [ ] **Expected**: Should convert to line art first, then segment

#### Test 13: With Stained Glass Effect
- [ ] Upload an image
- [ ] Enable **ML Segmentation**
- [ ] Enable **Stained Glass Effect**
- [ ] Process the image
- [ ] **Expected**: Should apply both ML segmentation and stained glass

#### Test 14: With Different Color Palettes
- [ ] Upload an image
- [ ] Enable **ML Segmentation**
- [ ] Test with different palettes (Vibrant, Ocean, Sunset, etc.)
- [ ] **Expected**: All palettes should work with ML segmentation

### ✅ Performance Tests

#### Test 15: Processing Time
- [ ] Upload a medium image (1000x1000px)
- [ ] Test processing time with ML **OFF**
- [ ] Test processing time with ML **ON** (SLIC)
- [ ] **Expected**: ML may be slightly slower but should complete within timeout

#### Test 16: Memory Usage
- [ ] Monitor memory usage during processing
- [ ] Test with large image and ML enabled
- [ ] **Expected**: Should not cause memory errors

### ✅ Regression Tests

#### Test 17: Traditional Method Still Works
- [ ] Upload various image types
- [ ] Keep ML segmentation **OFF**
- [ ] **Expected**: All existing functionality should work as before

#### Test 18: Settings Persistence
- [ ] Set ML segmentation **ON**, method: **SLIC**, regions: **100**
- [ ] Process an image
- [ ] Refresh the page
- [ ] **Expected**: Settings should reset to defaults (not persisted - this is expected)

## Success Criteria

Phase 4.1 ML Segmentation is complete when:

1. ✅ All UI components render correctly
2. ✅ ML segmentation toggle works (enables/disables features)
3. ✅ Method picker allows selection of auto/slic/edge
4. ✅ Target regions slider adjusts the number of regions
5. ✅ SLIC method produces multiple distinct regions (not mostly one color)
6. ✅ Edge method works for line art
7. ✅ Auto method selects appropriate technique
8. ✅ No crashes or errors during processing
9. ✅ Traditional method (ML OFF) still works as before
10. ✅ Integration with other features (line art, stained glass, palettes) works

## Known Issues to Watch For

1. **Single Color Output**: If ML segmentation produces mostly one color, try:
   - Increasing target regions (100-150)
   - Using traditional method (ML OFF) for photos with gradients
   - Using SLIC method with higher target regions

2. **Too Few Regions**: If not enough regions are detected:
   - Increase target regions slider
   - Check that merging threshold is conservative (should be 12.0)

3. **Processing Timeout**: If processing takes too long:
   - Reduce target regions
   - Use traditional method for very large images

## Next Steps After Testing

Once Phase 4.1 testing is complete:
- Update roadmap to mark Phase 4.1 as complete
- Document any issues found
- Consider Phase 3 features (Performance Optimizations, Advanced Adjacency Detection, Batch Processing)

