# TableSegmentation

Python system for segmenting table-based forms and census records, designed by Brian Robinson as a CS Master's project.
The system returns output as a JSON file of cell corners that can be sent to the RANSAC segmenter for further use.
Skewed images must be deskewed to get good results.

Package Requirements: numpy, matplotlib

Usage: `SegmentTable.py imgpath [-o OUTPATH] [-i OUTIMG] [-t TEMPLATE] [-f FILTER{0,1}] [-r ROWS]`

*imgpath*: Image file or directory of images to segment

*outpath*: Directory for the JSON output (defaults to imgpath)

*outimg*: (Optional) Directory for image output displaying segmentation results

*template*: (Optional) Template JSON containing good results for a similar image (In development)

*filter*: Enable filtering of output points (defaults to 1)

*rows*: Expected number of rows in the output (defauls to 51)

`run.py` can be used to run the segmenter without using the command line.

`deskewimg.py` can be used to deskew images before segmentation.
