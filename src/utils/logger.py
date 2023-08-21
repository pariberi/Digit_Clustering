from time import time
class Logger:
    @staticmethod
    def time_logger(func):
        def inner_func(self,*args):
            start_time = time()
            label = func(self, *args)
            end_time = time()
            total_time = int(end_time-start_time)
            min = int(total_time/60)
            sec= total_time % 60
            print(f'time: {min} minutes , {sec} seconds')
            return label
        return inner_func