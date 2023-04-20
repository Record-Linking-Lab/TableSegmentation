import cv2, os, json, argparse, traceback
import numpy as np
from matplotlib import pyplot as plt

def bin_lines(lines, bin_axis, bin_threshold, max_dist):
    lines = sorted(lines, key=lambda x: x[0][bin_axis])
    bins = []
    current_bin = [lines[0]]
    for l in lines[1:]:
        if l[0][bin_axis] - current_bin[-1][0][bin_axis] > bin_threshold or l[0][bin_axis] - current_bin[-1][0][bin_axis] > max_dist:
            bins.append(sorted(current_bin, key=lambda x: x[0][1 - bin_axis]))
            current_bin = [l]
        else:
            current_bin.append(l)
    bins.append(current_bin)
    return bins

def filter_lines(vals, filter_threshold):
    return [i for i,_ in enumerate(vals) if vals[i] > filter_threshold]

def filter_lines_avg(avgs, min_size):
    return [i for i,_ in enumerate(avgs) if abs(avgs[i] - avgs[i-1]) > min_size]

def filter_crosslines(lines, avgs, crosslines, axis, filter_threshold=30):
    lineind = [i for i,_ in enumerate(lines)]
    i = 0
    while i < len(lines)-1:
        i += 1
        avg = (avgs[i]+avgs[i-1])/2
        cross = [l for l in crosslines if (l[0][axis] - avg) * (l[0][axis+2] - avg) < 0]
        if len(cross) < filter_threshold:
            if i-1 not in lineind: return []
            lineind.remove(i-1)
        else:
            break
    i = len(lines)-1
    while i > 0:
        i -= 1
        avg = (avgs[i]+avgs[i+1])/2
        cross = [l for l in crosslines if (l[0][axis] - avg) * (l[0][axis+2] - avg) < 0]
        if len(cross) < filter_threshold:
            if i+1 not in lineind: return []
            lineind.remove(i+1)
        else:
            break
    return lineind

def interpolate_rows(avgH, bin_threshold, tightness=1):
    diff = sorted([avgH[i+1] - avgH[i] for i in range(len(avgH) - 1)])
    bins = []
    current_bin = [diff[0]]
    for v in diff[1:]:
        if v - current_bin[0] > bin_threshold:
            bins.append(current_bin)
            current_bin = [v]
        else:
            current_bin.append(v)
    bins.append(current_bin)
    bins = [b for b in bins if len(b) > 3]
    median = sum(bins[0]) / len(bins[0]) if bins else diff[0]
    y = 0
    limit = len(avgH)*20
    while y < len(avgH)-1 and y < limit:
        v1, v2 = avgH[y], avgH[y+1]
        diff = v2 - v1
        match = [999999]
        for i in range(1, 10):
            match.append(abs(diff - median*i))
        best = min(match)
        if best < median/tightness:
            besti = match.index(best)
            if besti > 1:
                for i in range(1, besti):
                    avgH.insert(y+i, (v1*i + v2*(besti-i)) / besti)
            y += besti
        else: y += 1
    return avgH

def line_slope(l):
    if l[0,0] == l[0,2]: return 99999999
    return (l[0,3] - l[0,1])/(l[0,2] - l[0,0])

def line_len_h(l):
    return abs((l[0,2] - l[0,0]))

def line_len_v(l):
    return abs((l[0,3] - l[0,1]))

def evaluate_match(vals1, vals2, offset):
    if offset > 0:
        vals1 = vals1[offset:]
    elif offset < 0:
        vals2 = vals2[-offset:]
    if len(vals1) > len(vals2):
        vals1 = vals1[:len(vals2)]
    elif len(vals2) > len(vals1):
        vals2 = vals2[:len(vals1)]
    
    diff = vals2[0] - vals1[0]
    vals1 = [v + diff for v in vals1]
    score = 0
    for i, s in enumerate(vals1):
        closest_dist = 99999999
        for i2, s2 in enumerate(vals2):
            if abs(s - s2) < closest_dist:
                closest_dist = abs(s - s2)
        score += closest_dist
    return score / len(vals1), diff

def get_positions(seg, temp):
    positions = []
    for i, v in enumerate(temp):
        for i2, v2 in enumerate(seg):
            if v2 >= v:
                v0 = seg[i2-1]
                positions.append(i2 + (v - v0)/(v2 - v0) - 1)
                break
    return positions

def get_points_from_pos(seg, pos, di):
    di = 0
    pos = [p + di for p in pos]
    pos = [p for p in pos if p >= 0 and p < len(seg)]
    return [seg[int(p)] for p in pos]

def align_with_template(img, avgH, avgV, segRows, segCols, templateRows, templateCols):
    BIN_THRESHOLD = 0.002
    BIN_MAX_DIST = BIN_THRESHOLD*3
    best_score = 9999999
    best_di = 0
    posH = get_positions(segRows, templateRows)
    for di in range(-3, 4):
        score, diff = evaluate_match(segRows, avgH, di, BIN_MAX_DIST*min(img.shape[0], img.shape[1]))
        if score < best_score:
            best_score = score
            best_di = di
    best_rows = get_points_from_pos(avgH, posH, best_di)
    best_score = 0
    best_cols = 0
    posV = get_positions(segCols, templateCols)
    for di in range(-3, 4):
        score, diff = evaluate_match(segCols, avgV, di, BIN_MAX_DIST*min(img.shape[0], img.shape[1]))
        if score < best_score:
            best_score = score
            best_cols = diff
    best_cols = get_points_from_pos(avgV, posV, best_di)
    return best_rows, best_cols

def deskew_img(img, linesH):
    avgS = sum([line_slope(l) for l in linesH])/len(linesH)
    print(avgS)
    angleR = -np.arctan(avgS)
    print(angleR)
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angleR, scale=1)
    return cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))

def segment(img, template=None, use_filter=True, filter_dist_percent=0.2):
    MAX_SIZE = 3000
    BIN_THRESHOLD = 0.002
    BIN_MAX_DIST = BIN_THRESHOLD*3
    SLOPE_THRESHOLD = 0.1
    WIDTH_THRESHOLD = 0.02
    FILTER_THRESHOLD = 0.1
    MIN_SIZE_PERCENT = 0.005
    scale = MAX_SIZE / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    filter_dist = int(img.shape[1]*filter_dist_percent/100)
    lsd = cv2.createLineSegmentDetector()
    
    #Detect lines in the image
    resultx = cv2.filter2D(src=img, ddepth=-1, kernel=np.ones((1, filter_dist), np.float32)/filter_dist)
    linesh = lsd.detect(resultx)[0]
    linesh = np.array([np.rint(l) for l in linesh if abs(line_slope(l)) < SLOPE_THRESHOLD
                      and line_len_h(l)/img.shape[1] > WIDTH_THRESHOLD/2])
    
    resulty = cv2.filter2D(src=img, ddepth=-1, kernel=np.ones((filter_dist, 1), np.float32)/filter_dist)
    linesv = lsd.detect(resulty)[0]
    linesv = np.array([np.rint(l) for l in linesv if abs(line_slope(l)) > 1/SLOPE_THRESHOLD
                      and line_len_v(l)/img.shape[0] > WIDTH_THRESHOLD])

    #Bin lines
    
    binsH = bin_lines(linesh, 1, BIN_THRESHOLD*min(img.shape[0], img.shape[1]), BIN_MAX_DIST*min(img.shape[0], img.shape[1]))
    sumH = [min(sum([abs(b[0][2] - b[0][0]) for b in r])/4, img.shape[1]*0.99) for r in binsH]
    avgH = [np.median([b[i][0][1] for i in range(len(b))]) for b in binsH]
    
    binsV = bin_lines(linesv, 0, BIN_THRESHOLD*min(img.shape[0], img.shape[1]), BIN_MAX_DIST*min(img.shape[0], img.shape[1]))
    sumV = [min(sum([abs(b[0][3] - b[0][1]) for b in r])/4, img.shape[0]*0.99) for r in binsV]
    avgV = [np.median([b[i][0][0] for i in range(len(b))]) for b in binsV]
    
    if use_filter:
        #Filter by row/column line sum
    
        ind = filter_lines(sumH, FILTER_THRESHOLD*img.shape[1])
        binsH = [binsH[i] for i in ind]
        avgH = [avgH[i] for i in ind]
        
        ind = filter_lines(sumV, FILTER_THRESHOLD*img.shape[0])
        binsV = [binsV[i] for i in ind]
        avgV = [avgV[i] for i in ind]
        
        #Filter by distance between lines
        
        ind = filter_lines_avg(avgH, MIN_SIZE_PERCENT*img.shape[1])
        binsH = [binsH[i] for i in ind]
        avgH = [avgH[i] for i in ind]
        
        ind = filter_lines_avg(avgV, MIN_SIZE_PERCENT*img.shape[0])
        binsV = [binsV[i] for i in ind]
        avgV = [avgV[i] for i in ind]
        
        #Filter using crosslines

        ind = filter_crosslines(binsH, avgH, sum(binsV, []), 1)
        if not ind: ind = filter_crosslines(binsH, avgH, sum(binsV, []), 1, filter_threshold=10)
        binsH = [binsH[i] for i in ind]
        avgH = [avgH[i] for i in ind]
        
        ind = filter_crosslines(binsV, avgV, sum(binsH, []), 0)
        if not ind: ind = filter_crosslines(binsV, avgV, sum(binsH, []), 0, filter_threshold=10)
        binsV = [binsV[i] for i in ind]
        avgV = [avgV[i] for i in ind]
        
        #Interpolate missing rows
        avgH = interpolate_rows(avgH, BIN_THRESHOLD*min(img.shape[0], img.shape[1]))
        
        rows = [int(v / scale) for v in avgH]
        cols = [int(v / scale) for v in avgV]
    
    if template:
        data = template
        segH, segV = data['segH'], data['segV']
        targetH, targetV = data['targetH'], data['targetV']
        alignH, alignV = align_with_template(img, rows, cols, segH, segV, targetH, targetV)
        return alignH, alignV
    return rows, cols

def export_seg(img, filename, seg_rows, seg_cols, target_rows, target_cols, outimg=''):
    seg_rows = [int(v) for v in seg_rows]
    seg_cols = [int(v) for v in seg_cols]
    target_rows = [int(v) for v in target_rows]
    target_cols = [int(v) for v in target_cols]
    jsonout = {'corners':[],'segH':seg_rows,'segV':seg_cols,'targetH':target_rows,'targetV':target_cols}
    for r in target_rows:
        for c in target_cols:
            jsonout['corners'].append([r, c])
    outfilename = filename.split('.')[0]+'.json'
    with open(outfilename, 'w') as outfile:
        json.dump(jsonout, outfile)
    print(len(jsonout['corners']), 'corners succesfully written to '+ outfilename)
    if outimg:
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, cmap='gray', origin='upper')
        v1, v2 = target_cols, target_rows
        ax.scatter(sum([[v]*len(v2) for v in v1], start=[]), v2*len(v1), s=2, c=[(1,0,0)])
        fig.savefig(outimg, dpi=500)
        print('Output image written to '+outimg)
        plt.close()

def display_cells(img, rows, cols):
    for i2,x in enumerate(cols[:-1]):
        for i,y in enumerate(rows[:-1]):
            x1, x2, y1, y2 = int(rows[i]), int(rows[i+1]), int(cols[i2]), int(cols[i2+1])
            plt.imshow(img[x1:x2, y1:y2], cmap="gray")
            plt.show()

def mainf(imgpath, outpath, outimg, template, filterlvl, exprows, data=''):
    if template and not data:
        if not template.endswith('.json'):
            template += '.json'
        with open(template) as f:
            data = json.load(f)
    if os.path.isdir(imgpath):
        for filename in os.listdir(imgpath):
            f = imgpath+'/'+filename
            if any([f.endswith(s) for s in ['.jpg', '.jpeg', '.png', '.bmp']]) and '-output' not in f:
                try:
                    mainf(f, outpath, outimg, template, filterlvl, exprows, data)
                except KeyboardInterrupt as e:
                    raise e
                except:
                    print('Exception thrown when segmenting '+f+':')
                    traceback.print_exc()
        return
    img = cv2.imread(imgpath, 0).astype('uint8')
    if img is None:
        print('Image/directory not found:', imgpath)
        return
    avgH, avgV = segment(img, data, use_filter=filterlvl)
    if template:
        trange = [0, -1, 0, -1]
    templateH = avgH[-exprows:]
    templateV = avgV
    filename = imgpath.split('/')[-1].split('.')[0]+'-output.jpg'
    if not outpath: outpath = '/'.join(imgpath.split('/')[:-1])
    if '.' not in outpath:
        if not outpath.endswith('/'): outpath += '/'
        outpath += filename
    if outimg and '.' not in outimg:
        if not outimg.endswith('/'): outimg += '/'
        outimg += filename
    export_seg(img, outpath, avgH, avgV, templateH, templateV, outimg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SegmentTable', description='Segment table image into cells', epilog='')
    parser.add_argument('imgpath')
    parser.add_argument('-o', '--outpath', required=False, default='')
    parser.add_argument('-i', '--outimg', required=False, default='')
    parser.add_argument('-t', '--template', required=False, default='')
    parser.add_argument('-f', '--filter', type=int, choices=[0, 1], required=False, default=1)
    parser.add_argument('-r', '--rows', type=int, required=False, default=51)
    args = parser.parse_args()
    mainf(args.imgpath, args.outpath, args.outimg, args.template, args.filter, args.rows)
    

