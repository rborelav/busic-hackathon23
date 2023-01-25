import time
import taichi as ti

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


# Import from Bunny DEM
class DEMSolverStatistics:
    class Timer:
        def __init__(self):
            self.first:bool = True
            self.on:bool = False
            self.start:float = 0.0
            self.total = 0.0

        def tick(self):
            ti.sync()
            if(self.on == False): 
                self.start = time.time()
                self.on = True
            else:
                if(self.first): self.first = False
                else: self.total += time.time() - self.start
                self.on = False
        
        def __str__(self):
            return str(self.total)
        
    def __init__(self):
        self.SolveTime = self.Timer()
        
        self.BroadPhaseDetectionTime = self.Timer()
        self.HashTableSetupTime = self.Timer()
        self.PrefixSumTime = self.Timer()
        self.CollisionPairSetupTime = self.Timer()
        
        self.ContactResolveTime = self.Timer()
        self.ContactTime = self.Timer()
        self.ResolveWallTime = self.Timer()
        self.ApplyForceTime = self.Timer()
        self.Apply_bcTime = self.Timer()
        self.UpdateTime = self.Timer()
        
    
    def _pct(self, x:Timer):
        if(self.SolveTime.total == 0.0): return '0%'
        return str(x.total / self.SolveTime.total * 100) + '%'
    
    def report(self):
        print(f"Total              = {self.SolveTime}\n"
            #   f"ApplyForceTime     = {self.ApplyForceTime}({self._pct(self.ApplyForceTime)})\n"
              f"UpdateTime         = {self.UpdateTime}({self._pct(self.UpdateTime)})\n"
            #   f"ResolveWallTime    = {self.ResolveWallTime}({self._pct(self.ResolveWallTime)})\n"
              f"ContactTime        = {self.ContactTime}({self._pct(self.ContactTime)})\n"
              f"Apply_bcTime        = {self.Apply_bcTime}({self._pct(self.Apply_bcTime)})\n"
            #   f"    -BPCD               = {self.BroadPhaseDetectionTime}({self._pct(self.BroadPhaseDetectionTime)})\n"
            #   f"        --HashTableSetupTime      = {self.HashTableSetupTime}({self._pct(self.HashTableSetupTime)})\n"
            #   f"        --PrefixSumTime           = {self.PrefixSumTime}({self._pct(self.PrefixSumTime)})\n"
            #   f"        --CollisionPairSetupTime  = {self.CollisionPairSetupTime}({self._pct(self.CollisionPairSetupTime)})\n"
            #   f"    -ContactResolveTime = {self.ContactResolveTime}({self._pct(self.ContactResolveTime)})\n"
              )

    def report_avg(self, step):
        print(
            #   f"ApplyForceTime     = {self.ApplyForceTime}({self._pct(self.ApplyForceTime)})\n"
              f"UpdateTime         = {self.UpdateTime.total/step}({self._pct(self.UpdateTime)})\n"
            #   f"ResolveWallTime    = {self.ResolveWallTime}({self._pct(self.ResolveWallTime)})\n"
              f"ContactTime        = {self.ContactTime.total/step}({self._pct(self.ContactTime)})\n"
              f"Apply_bcTime        = {self.Apply_bcTime.total/step}({self._pct(self.Apply_bcTime)})\n"
            #   f"    -BPCD               = {self.BroadPhaseDetectionTime}({self._pct(self.BroadPhaseDetectionTime)})\n"
            #   f"        --HashTableSetupTime      = {self.HashTableSetupTime}({self._pct(self.HashTableSetupTime)})\n"
            #   f"        --PrefixSumTime           = {self.PrefixSumTime}({self._pct(self.PrefixSumTime)})\n"
            #   f"        --CollisionPairSetupTime  = {self.CollisionPairSetupTime}({self._pct(self.CollisionPairSetupTime)})\n"
            #   f"    -ContactResolveTime = {self.ContactResolveTime}({self._pct(self.ContactResolveTime)})\n"
              )
