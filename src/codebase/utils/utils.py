def log(fh, msg):
    '''
    Log a message and print it.
    '''
    print(msg)
    if msg[-1] != '\n':
        msg += '\n'
    fh.write(msg)
