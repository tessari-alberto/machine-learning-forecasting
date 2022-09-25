from datetime import datetime
import winsound

def parse_date_data1(date: str):
	return datetime.strptime(date, '')


def date_from_year(year: str):
	return datetime.strptime(year, '%Y')

def ring_bell():
	duration = 500  # milliseconds
	freq = 300 #440  # Hz
	winsound.Beep(freq, duration)
	winsound.Beep(freq, duration)


def same_line_print(msg):
    print('\t' + msg, sep=' ', end='', flush=True)


# PROGRESS BAR

class ProgressBar:

    def __init__(self, length=10):
        self.progression_bar_length = length

    def set_length(self, length):
        self.progression_bar_length = length

    def get_length(self, ):
        return self.progression_bar_length

    # update_progress() : Displays or updates a console progress bar
    # Accepts a float between 0 and 1. Any int will be converted to a float.
    # A value under 0 represents a 'halt'.
    # A value at 1 or bigger represents 100%
    def progress(self, progress, estimated_remaining_time='', time_unit='s'):
        barLength = self.progression_bar_length  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        if status == '':
            status = '\t' + str(estimated_remaining_time) + time_unit + ' remaining'
        text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100),
                                                  status)
        # text = "\rExecuting clustering: [{0}]".format( "#"*block + "-"*(barLength-block))
        sys.stdout.write(text)
        sys.stdout.flush()