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

        var projectRoot = Directory.GetParent(assetsDir) ?? new DirectoryInfo(assetsDir);
        var outputDir = Path.Combine(projectRoot.FullName, "Output");
        Directory.CreateDirectory(outputDir);

        var imageFiles = Directory.GetFiles(assetsDir)
            .Where(f => ImageExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToArray();

        if (imageFiles.Length == 0)
        {
            Console.WriteLine("No image files found in Assets folder.");
            Console.WriteLine($"Supported formats: {string.Join(", ", ImageExtensions)}");
            return;
        }

        Console.WriteLine("DetectImageEdges - Edge Detection Assessment\n");
        Console.WriteLine($"Found {imageFiles.Length} image file(s) to process.\n");

        var processedCount = 0;
        foreach (var inputPath in imageFiles)
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
                continue;
            }

            using var edges = DetectEdges(source);
            Cv2.ImWrite(destination, edges);
            Console.WriteLine($" Saved edge map to '{destination}'.\n");

            processedCount++;
        }

        Console.WriteLine($"Done. Processed {processedCount} image(s). Inspect the Output folder for the generated PNG files.");
    }

    private static Mat DetectEdges(Mat source)
    {
        using var prepared = PrepareSource(source);
        using var gray = new Mat();
        Cv2.CvtColor(prepared, gray, ColorConversionCodes.BGR2GRAY);

        using var blurred = new Mat();
        Cv2.GaussianBlur(gray, blurred, new Size(7, 7), 2.0);

        using var mask = BuildMask(blurred);
        var contours = FindMainContours(mask, source.Width * source.Height);

        var output = new Mat(source.Size(), MatType.CV_8UC3, Scalar.All(255));
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

        return input.Clone();
    }

    private static Mat BuildMask(Mat gray)
    {
        var (mean, stdDev) = MeasureBackground(gray);

        using var difference = new Mat();
        Cv2.Absdiff(gray, new Scalar(mean), difference);

        var brightBackground = mean > 180;
        var contrastThreshold = brightBackground ? 5 : Math.Max(10, stdDev);

        using var contrastMask = new Mat();
        Cv2.Threshold(difference, contrastMask, contrastThreshold, 255, ThresholdTypes.Binary);

        using var otsuMask = new Mat();
        Cv2.Threshold(gray, otsuMask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

        var mask = new Mat();
        Cv2.BitwiseAnd(contrastMask, otsuMask, mask);

        var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel, iterations: 2);
        Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernel, iterations: 1);

        return mask;
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

        if (samples.Count == 0)
            return (127, 10);

        var mean = samples.Average(v => (double)v);
        var variance = samples.Average(v => Math.Pow(v - mean, 2));
        return (mean, Math.Sqrt(variance));
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