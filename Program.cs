using OpenCvSharp;

internal class Program
{
    private static readonly string[] ImageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp" };

    private static void Main(string[] args)
    {
        var assetsDir = FindAssetsFolder();
        if (assetsDir is null)
        {
            Console.WriteLine("Assets folder not found. Place input images inside an 'Assets' directory.");
            return;
        }

        var outputDir = SetupOutputDirectory(assetsDir);
        var imageFiles = GetImageFiles(assetsDir);

        if (imageFiles.Length == 0)
        {
            Console.WriteLine("No image files found in Assets folder.");
            Console.WriteLine($"Supported formats: {string.Join(", ", ImageExtensions)}");
            return;
        }

        PrintHeader(imageFiles.Length);
        var processedCount = ProcessImages(imageFiles, outputDir);
        PrintSummary(processedCount);
    }

    private static string SetupOutputDirectory(string assetsDir)
    {
        var projectRoot = Directory.GetParent(assetsDir) ?? new DirectoryInfo(assetsDir);
        var outputDir = Path.Combine(projectRoot.FullName, "Output");
        Directory.CreateDirectory(outputDir);
        return outputDir;
    }

    private static string[] GetImageFiles(string assetsDir)
    {
        return Directory.GetFiles(assetsDir)
            .Where(f => ImageExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToArray();
    }

    private static void PrintHeader(int imageCount)
    {
        Console.WriteLine("DetectImageEdges - Edge Detection Assessment\n");
        Console.WriteLine($"Found {imageCount} image file(s) to process.\n");
    }

    private static int ProcessImages(string[] imageFiles, string outputDir)
    {
        var processedCount = 0;
        foreach (var inputPath in imageFiles)
        {
            if (ProcessSingleImage(inputPath, outputDir))
            {
                processedCount++;
            }
        }
        return processedCount;
    }

    private static bool ProcessSingleImage(string inputPath, string outputDir)
    {
        var fileName = Path.GetFileName(inputPath);
        var nameWithoutExt = Path.GetFileNameWithoutExtension(inputPath);
        var outputFileName = $"output-{nameWithoutExt}.png";
        var destination = Path.Combine(outputDir, outputFileName);

        Console.WriteLine($"Processing: {fileName}...");

        using var source = Cv2.ImRead(inputPath, ImreadModes.Color);
        if (source.Empty())
        {
            Console.WriteLine($" Skipping {fileName}: failed to load image.\n");
            return false;
        }

        using var edges = DetectEdges(source);
        Cv2.ImWrite(destination, edges);
        Console.WriteLine($" Saved edge map to '{destination}'.");

        PrintPreview(edges);
        return true;
    }

    private static void PrintPreview(Mat edges)
    {
        Console.WriteLine("Preview:");
        Console.WriteLine(RenderAscii(edges, maxWidth: 72));
        Console.WriteLine();
    }

    private static void PrintSummary(int processedCount)
    {
        Console.WriteLine($"Done. Processed {processedCount} image(s). Inspect the Output folder for the generated PNG files.");
    }

    private static Mat DetectEdges(Mat source)
    {
        using var prepared = PrepareSource(source);
        using var gray = ConvertToGrayscale(prepared);
        using var blurred = ApplyGaussianBlur(gray);
        using var mask = BuildMask(blurred);
        var contours = FindMainContours(mask, source.Width * source.Height);

        return DrawContoursOnCanvas(source.Size(), contours);
    }

    private static Mat ConvertToGrayscale(Mat source)
    {
        var gray = new Mat();
        Cv2.CvtColor(source, gray, ColorConversionCodes.BGR2GRAY);
        return gray;
    }

    private static Mat ApplyGaussianBlur(Mat gray)
    {
        var blurred = new Mat();
        Cv2.GaussianBlur(gray, blurred, new Size(7, 7), 2.0);
        return blurred;
    }

    private static Mat DrawContoursOnCanvas(Size imageSize, Point[][] contours)
    {
        var output = new Mat(imageSize, MatType.CV_8UC3, Scalar.All(255));

        if (contours.Length > 0)
        {
            foreach (var contour in contours)
            {
                var smoothed = Cv2.ApproxPolyDP(contour, 2.0, true);
                Cv2.DrawContours(output, new[] { smoothed }, -1, new Scalar(255, 0, 0), thickness: 3, lineType: LineTypes.AntiAlias);
            }
        }

        return output;
    }

    private static Mat PrepareSource(Mat input)
    {
        if (input.Channels() == 4)
        {
            return HandleAlphaChannel(input);
        }

        return input.Clone();
    }

    private static Mat HandleAlphaChannel(Mat input)
    {
        using var bgr = new Mat();
        Cv2.CvtColor(input, bgr, ColorConversionCodes.BGRA2BGR);

        var composed = new Mat(bgr.Size(), MatType.CV_8UC3, Scalar.All(255));
        Cv2.Split(input, out var channels);
        using var alpha = channels[3];
        bgr.CopyTo(composed, alpha);

        foreach (var channel in channels)
        {
            channel.Dispose();
        }

        return composed;
    }

    private static Mat BuildMask(Mat gray)
    {
        var (mean, stdDev) = MeasureBackground(gray);
        using var contrastMask = CreateContrastMask(gray, mean, stdDev);
        using var otsuMask = CreateOtsuMask(gray);
        
        var mask = CombineMasks(contrastMask, otsuMask);
        ApplyMorphology(mask);
        
        return mask;
    }

    private static Mat CreateContrastMask(Mat gray, double mean, double stdDev)
    {
        using var difference = new Mat();
        Cv2.Absdiff(gray, new Scalar(mean), difference);

        var brightBackground = mean > 180;
        var contrastThreshold = brightBackground ? 5 : Math.Max(10, stdDev);

        var contrastMask = new Mat();
        Cv2.Threshold(difference, contrastMask, contrastThreshold, 255, ThresholdTypes.Binary);
        return contrastMask;
    }

    private static Mat CreateOtsuMask(Mat gray)
    {
        var otsuMask = new Mat();
        Cv2.Threshold(gray, otsuMask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        return otsuMask;
    }

    private static Mat CombineMasks(Mat contrastMask, Mat otsuMask)
    {
        var mask = new Mat();
        Cv2.BitwiseAnd(contrastMask, otsuMask, mask);
        return mask;
    }

    private static void ApplyMorphology(Mat mask)
    {
        var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel, iterations: 2);
        Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernel, iterations: 1);
    }

    private static Point[][] FindMainContours(Mat mask, int imageArea)
    {
        Cv2.FindContours(mask, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        if (contours.Length == 0)
            return Array.Empty<Point[]>();

        var minArea = imageArea * 0.001;

        return contours
            .Select(c => (Contour: c, Area: Cv2.ContourArea(c)))
            .Where(t => t.Area >= minArea)
            .OrderByDescending(t => t.Area)
            .Take(3)
            .Select(t => t.Contour)
            .ToArray();
    }

    private static (double Mean, double StdDev) MeasureBackground(Mat gray)
    {
        var borderSamples = CollectBorderSamples(gray);
        
        if (borderSamples.Count == 0)
            return (127, 10);

        return CalculateStatistics(borderSamples);
    }

    private static List<byte> CollectBorderSamples(Mat gray)
    {
        var rows = gray.Rows;
        var cols = gray.Cols;
        var border = Math.Max(4, Math.Min(rows, cols) / 20);

        var samples = new List<byte>();
        for (var y = 0; y < rows; y++)
        {
            for (var x = 0; x < cols; x++)
            {
                var isBorder = y < border || y >= rows - border || x < border || x >= cols - border;
                if (isBorder)
                {
                    samples.Add(gray.At<byte>(y, x));
                }
            }
        }

        return samples;
    }

    private static (double Mean, double StdDev) CalculateStatistics(List<byte> samples)
    {
        var mean = samples.Average(v => (double)v);
        var variance = samples.Average(v => Math.Pow(v - mean, 2));
        return (mean, Math.Sqrt(variance));
    }

    private static string RenderAscii(Mat image, int maxWidth)
    {
        if (maxWidth <= 0)
            return string.Empty;

        using var resized = ResizeForAscii(image, maxWidth);
        return ConvertToAsciiArt(resized);
    }

    private static Mat ResizeForAscii(Mat image, int maxWidth)
    {
        var targetWidth = Math.Min(maxWidth, image.Width);
        var aspectRatio = (double)image.Height / image.Width;
        var targetHeight = Math.Max(1, (int)Math.Round(targetWidth * aspectRatio * 0.5));

        var resized = new Mat();
        Cv2.Resize(image, resized, new Size(targetWidth, targetHeight), 0, 0, InterpolationFlags.Area);
        return resized;
    }

    private static string ConvertToAsciiArt(Mat resized)
    {
        const string ramp = " .:-=+*#%@";
        var builder = new System.Text.StringBuilder();

        for (var y = 0; y < resized.Height; y++)
        {
            for (var x = 0; x < resized.Width; x++)
            {
                var pixel = resized.At<Vec3b>(y, x);
                var b = pixel.Item0;
                var g = pixel.Item1;
                var r = pixel.Item2;

                var character = GetAsciiCharacter(r, g, b, ramp);
                builder.Append(character);
            }
            builder.AppendLine();
        }

        return builder.ToString();
    }

    private static char GetAsciiCharacter(byte r, byte g, byte b, string ramp)
    {
        if (r > 200 && g < 50 && b < 50)
        {
            return '@';
        }

        var brightness = (r + g + b) / (3.0 * 255.0);
        var index = (int)Math.Clamp(brightness * (ramp.Length - 1), 0, ramp.Length - 1);
        return ramp[index];
    }

    private static string? FindAssetsFolder()
    {
        foreach (var root in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var directory = new DirectoryInfo(root);
            while (directory is not null)
            {
                var candidate = Path.Combine(directory.FullName, "Assets");
                if (Directory.Exists(candidate))
                {
                    return candidate;
                }

                directory = directory.Parent;
            }
        }

        return null;
    }
}