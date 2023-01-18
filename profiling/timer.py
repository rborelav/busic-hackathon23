import time

class Timer:
    def __init__(self):
        self.logger = {}

    def start(self, func_name):
        if not func_name in self.logger:
            self.logger[func_name] = {'start_time':time.perf_counter(), 'exec_count':0, 'cum_runtime':0}
        else:
            self.logger[func_name]['start_time'] = time.perf_counter()

    def log(self, func_name):
        if self.logger[func_name]:
            self.logger[func_name]['exec_count'] += 1
            self.logger[func_name]['cum_runtime'] += time.perf_counter() - self.logger[func_name]['start_time']

    def output_log(self, filename):
        total_time = sum([val['cum_runtime'] for val in self.logger.values()])
        outfile = open(filename,'w')
        outfile.write('function\tncalls\ttottime\tpercall\treltime\n')
        for key,value in self.logger.items():
            outdata = '{}\t{}\t{}\t{}\t{:.1f}\n'.format(key,
                                                    value['exec_count'],
                                                    value['cum_runtime'],
                                                    value['cum_runtime']/value['exec_count'],
                                                    100*value['cum_runtime']/total_time)
            outfile.write(outdata)
        outfile.close()
