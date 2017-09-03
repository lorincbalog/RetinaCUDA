import subprocess
import time

if __name__ == '__main__':
    '''
    for f in ['speedup', 'ideal']:
        for c in ['False', 'True']:
            for r in range(0,4):
                for i in range(0,6):
		    if f == 'speedup' : stop = '50'
                    else: stop = '500'
                    test = 'python perf_eval.py %s %s %s %s %s %s' % (f, r, i, stop, 'True', c)
                    print test
                    proc = subprocess.Popen(test, shell=True)
                    proc.wait()
                    time.sleep(20)         
    '''
    for c in ['False', 'True']:
        for i in range(0,6):
            test = 'python eval50k.py %s %s %s %s' % (c, i, '500', 'True')
            print test
            proc = subprocess.Popen(test, shell=True)
            proc.wait()
            time.sleep(20)
