## DetectImageEdges

A .NET console application that processes input images and detects edges using OpenCV's Canny edge detection algorithm. The application generates output images containing only the detected edges (blue edges on white background) and displays ASCII previews in the console.

### Requirements

- .NET 8 SDK
- OpenCvSharp4.Windows (automatically installed via NuGet)

### Project Structure

- `Assets/` - Place your input images here (`input1.jpg`, `input2.png`)
- `Output/` - Generated edge detection results are saved here

### Usage

1. Place your input images in the `Assets/` folder (supports: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp)

2. Run the application:
   ```powershell
   dotnet run
   ```

3. The application will:
   - Automatically discover and process all image files in the `Assets/` folder
   - Detect edges using contour extraction
   - Generate output images with blue edges on white background
   - Save each output as `Output/output-{filename}.png`
   - Display ASCII previews of all outputs in the console

### Output

The output images contain only the detected edges, displayed as blue lines on a white background, matching the style of the provided reference examples.
