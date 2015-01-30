import time
import sys
from contextlib import contextmanager


def fmt_ival(t):
    '''Format and interval'''
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '%d:%02d:%02d' % (h, m, s)
    else:
        return '%02d:%02d' % (m, s)


class ProgressMeter(object):
    def __init__(self, total=0, description='', meter_length=15, disp=True):
        self.description = description
        self.meter_length = meter_length
        self.total = total
        self.curr = 0
        self.msg = ''
        self.disp = disp
        self.last_line_length = 0
        self.finished = False
        self.start_time = time.time()
        self._spinner = '-\|/-'
        self._spinner_idx = 0
        self._update()

    def spinner(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner)
        return self._spinner[self._spinner_idx]

    def set_message(self, msg):
        self.msg = msg
        self.print_progress()

    def increment(self, n=1):
        assert n > 0
        self.curr += n
        self._update()
        self.print_progress()

    def _update(self):
        ratio = 0 if self.total == 0 else self.curr / float(self.total)
        if ratio > 1:
            completed_length = self.meter_length
            remaining_length = 0
        else:
            completed_length = int(ratio * self.meter_length)
            remaining_length = (self.meter_length - completed_length)
        elapsed = time.time() - self.start_time
       
        left = '?'
        diff = self.total - self.curr
        if self.curr != 0 and self.total != 0 and diff >= 0:
            left = fmt_ival(elapsed / self.curr * diff)
        rate = '?' if elapsed == 0 else '%4.1f' % (self.curr / elapsed)

        msg = '\r'
        if self.description:
            msg += self.description + ': '
        if self.total != 0:
            msg += '[%s]' % ('#' * completed_length + 
                             self.spinner() * (remaining_length > 0) +
                             '-' * (remaining_length - 1))
            msg += ' %4.1f%%' % (100 * ratio)
        else:
            msg += '[%s]' % self.spinner()
        msg += '  %d/%d ' % (self.curr, self.total)
        msg += ' [%s %s %s]' % (fmt_ival(elapsed), left, rate)
        self._progress_msg = msg

    def print_progress(self):
        if self.disp:
            msg = self._progress_msg
            msg += ' ' + self.msg
            msg += ' ' * max(self.last_line_length - len(msg), 0)
            sys.stdout.write(msg)
            sys.stdout.flush()
            self.last_line_length = len(msg)

    def finish(self):
        if self.disp and not self.finished:
            self.finished = True
            self.print_progress()
            sys.stdout.write('\n')
            sys.stdout.flush()


@contextmanager
def progress_meter_ctx(total=0, description='', disp=True):
    meter = ProgressMeter(total, description, disp=disp)
    try:
        yield meter
    finally:
        meter.finish()
