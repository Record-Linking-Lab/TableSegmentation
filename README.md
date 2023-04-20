# TableSegmentation

Python system for segmenting table-based forms and census records, designed by Brian Robinson as a CS Master's project.
The system returns output as a JSON file of cell corners that can be sent to the RANSAC segmenter for further use.
Skewed images must be deskewed to get high quality results.

Package Requirements: numpy, matplotlib

Usage: `SegmentTable.py imgpath [-o OUTPATH] [-i OUTIMG] [-t TEMPLATE] [-f FILTER{0,1}] [-r ROWS]`

*imgpath*: Image file or directory of images to segment <br>
*outpath*: Directory for the JSON output (defaults to imgpath) <br>
*outimg*: (Optional) Directory for image output displaying segmentation results <br>
*template*: (Optional) Template JSON containing good results for a similar image (In development) <br>
*filter*: Enable filtering of output points (defaults to 1) <br>
*rows*: Expected number of rows in the output (defauls to 51) <br>

`run.py` can be used to run the segmenter without using the command line.

`deskewimg.py` can be used to deskew images before segmentation.

<br>

**A suggested workflow to use the program is as follows:**

1.	Set up folders for output and a folder consisting of a small number of input images.

2.	Run `deskewimg.py` to straighten the images, if necessary.

3.	Count the number of rows in the table.

4.	Run the segmenter with `run.py` or `SegmentTable.py` on the input folder with the expected number of rows and with the option to output images of results enabled.

5.	Look at the results to determine if segmentation is of acceptable quality.

6.	If quality is inconsistent but one image is error-free, take the output JSON for that image and run the RANSAC segmentation algorithm on the full record collection using the JSON as the template.

7.	Alternatively, if all images have a high-quality segmentation, instead run this table segmentation algorithm on the full image collection (with image output disabled for faster results)
