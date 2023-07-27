import os, cv2, traceback
import numpy as np

imgpath = '/home/jordantreypatton/work/TableSegmentation/images_for_testing/Mexico/d1'
outpath = '/home/jordantreypatton/work/TableSegmentation/deskewedOutputImg'

def line_slope(l):
    if l[0,0] == l[0,2]: return 99999999
    return (l[0,3] - l[0,1])/(l[0,2] - l[0,0])

def line_len_h(l):
    return abs((l[0,2] - l[0,0]))

def deskew_img(imgpath, outpath):
    if os.path.isdir(imgpath):
        for filename in os.listdir(imgpath):
            f = imgpath+'/'+filename
            if any([f.endswith(s) for s in ['.jpg', '.jpeg', '.png', '.bmp']]) and '-output' not in f:
                try:
                    deskew_img(f, outpath)
                except KeyboardInterrupt as e:
                    raise e
                except:
                    print('Exception thrown when deskewing '+f+':')
                    traceback.print_exc()
        return
    img = cv2.imread(imgpath, 0).astype('uint8')
    if img is None:
        print('Image/directory not found:', imgpath)
        return
    height, width = img.shape[:2]
    center = (width/2, height/2)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(img)[0]
    linesf = [l for l in lines if abs(line_slope(l)) < 0.2 and line_len_h(l) > width/20]
    if not linesf: linesf = lines
    angleR = sum([np.arctan(line_slope(l)) for l in linesf])/len(linesf)
    angleR *= 180/3.14
    print(angleR, " degrees")
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angleR, scale=1)
    img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
    filename = imgpath.split('/')[-1]
    if not outpath: outpath = '/'.join(filename.split('/')[:-1])
    if '.' not in outpath:
        if not outpath.endswith('/'): outpath += '/'
        outpath += filename
    cv2.imwrite(outpath, img)
    print("Deskewed image written to " + outpath)

deskew_img(imgpath, outpath)