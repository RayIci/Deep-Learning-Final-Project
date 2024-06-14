class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

def print_color(msg, color):
    print(color + msg + bcolors.ENDC)
    
def info(msg):
    print(msg)

def success(msg):
    print_color(msg, bcolors.OKGREEN)

def warning(msg):
    print_color(msg, bcolors.WARNING)
    
def error(msg):
    print_color(msg, bcolors.FAIL)