import subprocess
import time

# # measure initialisation time, performance and Python difference
# for i in range(0,5):
#     for size in img_sizes[1:]:
#         print "Retina index: %i" % i
#         print size
#         print "Grayscale"
#         perf_test.speedup_cam(loc[i],coeff[i], size, 500, True, False)

# for i in range(0,5):
#     for size in img_sizes[1:]:
#         print "Retina index: %i" % i
#         print size
#         print "Color"
#         perf_test.speedup_cam(loc[i],coeff[i], size, 500, True, True)


# # measure raw performance in ideal envrionment (e.g. one initalisation, use object on a lot of data)
# for i in range(0,5):
#     for size in img_sizes:
#         print "Retina index: %i" % i
#         print size
#         print "Color"
#         # start nvprof

#         perf_test.ideal_usage_cam(loc[i],coeff[i], size, 500, True, False)

if __name__ == '__main__':
    for f in ['speedup', 'ideal']:
        for c in ['False', 'True']:
            for r in range(0,4):
                for i in range(0,6):
                    test = 'nvprof python perf_eval.py %s %s %s %s %s %s' % (f, r, i, '500', 'False', c)
                    print test
                    proc = subprocess.Popen(test, shell=True)
                    proc.wait()
                    time.sleep(20)         

    for c in ['False', 'True']
        for i in range(0,6):
            test = 'nvprof python eval50k.py %s %s %s %s' % (c, i, '500', 'False')
            proc = subprocess.Popen(test, shell=True)
            proc.wait()
            time.sleep(20)