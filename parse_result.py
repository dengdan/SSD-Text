import util
import sys
import glob

result_dir = util.io.get_absolute_path(sys.argv[1])
#result_txts = glob.glob(r'%s/*/*/multiscale_evalfixed.xml'%(result_dir))
result_txts = glob.glob(r'%s/eval/*/test/*/fixed_eval.xml'%(result_dir))

def get_iter(xml):
    data = util.str.find_all(xml, 'model.ckpt\-\d+')[0]
    iter = int(data.split('-')[-1])
    return iter
def get_neg_overlap(xml):
#    data = util.str.find_all(xml, 'neg_overlap_0\.\d+')[0]
#    overlap = float(data.split('_')[-1])
    return 0.25
def get_confidence(xml):
    data = util.str.find_all(xml, 'confidence_0\.\d+')[0]
    confidence = float(data.split('_')[-1])
    return confidence
def get_nms_threshold(xml):
    data = util.str.find_all(xml, 'nms_threshold_0\.\d+')[0]
    nms_threshold = float(data.split('_')[-1])
    return nms_threshold
def get_score(data):
    scores = util.str.find_all(data, '0\.+\d+')
    if len(scores) == 0:
        scores = [0.0] * 3;
    return [float(d) for d in scores[:3]]
def get_result_str(xml):
    content = util.io.cat(xml)
    data = util.str.find_all(content, '\<score.+\/\>')[0]
    return data[1:-2]
class Result(object):
    def __init__(self, xml):
        self.iteration = get_iter(xml)
        self.neg_overlap = get_neg_overlap(xml)
        self.confidence = get_confidence(xml)
        self.nms_threshold = get_nms_threshold(xml)
        self.data = get_result_str(xml)
        self.r, self.p, self.f = get_score(self.data)
    
    def Print(self):
        print "|%d|%f|%f|`%s`|"%(self.iteration,  self.confidence, self.nms_threshold, self.data)
def sort_by_iteration_confidence(r1, r2):
    if r1.iteration == r2.iteration:
        return -1 if r1.confidence < r2.confidence else 1
    else:
        return r1.iteration - r2.iteration
def sort_by_f(r1, r2):
    return -1 if r1.f > r2.f else 1
def sort_by_r(r1, r2):
    return -1 if r1.r > r2.r else 1
def sort_by_p(r1,r2):
    return -1 if r1.p > r2.p else 1
results = []
for xml in result_txts:
    result = Result(xml)
    #if result.confidence == 0.25:
#    if result.post_nms == "":
#        continue
#    if result.iteration == 1500:
    results.append(result)
results.sort(sort_by_f)

#print "|Iter|neg_overlap|confidence|nms_threshold|post_nms|result|"
print "|Iter|confidence|nms_threshold|result|"
print "|---|---|---|---|---|"
for r in results:
    r.Print()
