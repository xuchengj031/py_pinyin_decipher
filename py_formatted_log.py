import sys
import logging
import math

logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)
fname = "留言1-1_op_已加密_转写"
fh = logging.FileHandler('log/' + fname + '.log', 'w', encoding='utf-8')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def list_columns(obj, cols=4, columnwise=False, gap=4):
    """
    Print the given list in evenly-spaced columns.

    Parameters
    ----------
    obj : list
        The list to be printed.
    cols : int
        The number of columns in which the list should be printed.
    columnwise : bool, default=True
        If True, the items in the list will be printed column-wise.
        If False the items in the list will be printed row-wise.
    gap : int
        The number of spaces that should separate the longest column
        item/s from the next column. This is the effective spacing
        between columns based on the maximum len() of the list items.
    """

    sobj = [str(item) for item in obj]
    if cols > len(sobj):
        cols = len(sobj)
    max_len = max([len(item) for item in sobj] or [1])
    if columnwise:
        cols = int(math.ceil(float(len(sobj)) / float(cols)))
    plist = [sobj[i: i + cols] for i in range(0, len(sobj), cols or 1)]
    if columnwise:
        if not len(plist[-1]) == cols:
            plist[-1].extend([''] * (len(sobj) - len(plist[-1])))
        plist = zip(*plist)
    printer = '\n'.join([
        ''.join([c.ljust(max_len + gap) for c in p])
        for p in plist])
    return printer


def lineno():
    frame = None
    try:
        raise ZeroDivisionError
    except ZeroDivisionError:
        frame = sys.exc_info()[2].tb_frame.f_back
    return frame.f_lineno


def print_multicol(my_list):
    max_len = len(max(my_list, key=len)) + 2
    ncols = (int(cols) - 4) / max_len
    while my_list:
        n = 0
        while n < ncols:
            if len(my_list) > 0:
                fstring = "{:<" + str(max_len) + "}"
                print(fstring.format(my_list.pop(0)))
            n += 1


def strlistToColumns(strl, maxWidth=80, spacing=4):
    longest = max([len(s) for s in strl])
    width = longest + spacing

    # compute numCols s.t. (numCols-1)*(longest+spacing)+longest < maxWidth
    numCols = 1 + (maxWidth - longest) // width
    C = range(numCols)

    # If len(strl) does not have a multiple of numCols, pad it with empty strings
    length = len(strl) if len(strl) else 1
    numCols = numCols if numCols else 1
    print(len(strl),length,numCols)
    strl += [""] * (length % numCols)
    numRows = length / numCols
    colString = ''

    for r in range(math.ceil(numRows)):
        colString += "".join(["{" + str(c) + ":" + str(width) + "}"
                              for c in C] + ["\n"]).format(*(strl[numCols * r + c]
                                                             for c in C))

    return colString


def info(cont, title='', cols=1, mark=[[], []]):
    if type(cont) in (list,):
        infolist(cont, title, cols)
    elif type(cont) == dict:
        infodict(cont, title, cols)
    else:
        a = '' if not mark[0] else '{} -> {}'.format(str(mark[0]),str(mark[1]))
        a = '' if not mark[1] else a
        cont = str(cont)
        logger.info("-" * 72)
        logger.info(r">>> {} (Type:{},Len:{}) : {}".format(title,str(type(cont))[8:-2], len(str(cont)), a))
        logger.info("-" * 72)
        logger.info(cont)
        logger.info("-" * 72)
        logger.info("\n")


def infolist(cont, title='', cols=4, son_cols=4,gap=4):
    if type(cont) in [int,float,bool]:
        cont = str(cont)
    logger.info("-" * 72)
    logger.info(r">>> {} (Type:{},Len:{})".format(
        title, str(type(cont))[8:-2], len(cont)))
    logger.info("-" * 72)
    # if type(cont[0]) == list or type(cont[0]) == set:
    if type(cont[0]) == list:
        for k,v in enumerate(cont):
            if v:
                infolist_son(v, '{}[{}]'.format(title,k), son_cols)
    elif type(cont[0]) == set:
        for k,v in enumerate(cont):
            if v:
                infoset_son(v, '{}[{}]'.format(title,k), son_cols)
    else:
        logger.info(list_columns(cont, cols=cols,gap=gap))
    logger.info("-" * 72)
    logger.info("\n")

def infodict_son(cont, title='', cols=4,gap=4):
    logger.info(r"> {} (Type:{},Len:{})".format(title, str(type(cont))[8:-2], len(cont)))
    for i,j in cont.items():
        holder = ["{} -> {}".format(i, cont[i]) for i in cont.keys()]
        logger.info(list_columns(holder, cols=cols,gap=gap))
    logger.info("-" * 40)

def infolist_son(cont, title='', cols=4,gap=4):
    logger.info(r"> {} (Type:{},Len:{})".format(title, str(type(cont))[8:-2], len(cont)))
    if type(cont) == set:
        for i in cont:
            logger.info(list_columns(i,cols=cols,gap=gap))
    if type(cont) == list:
        # for k,v in enumerate(cont):
            # holder = ["[{}] {}".format(m, n) for m,n in enumerate(cont)]
            # holder = v
        logger.info(list_columns(cont, cols))
        logger.info("-" * 40)
def infoset_son(cont, title='', cols=1, son_cols=4,gap=4):
    logger.info(r"> {} (Type:{},Len:{})".format(title, str(type(cont))[8:-2], len(cont)))
    logger.info("-" * 40)
    logger.info(list_columns(cont, cols=cols,gap=gap))
    logger.info("-" * 40)

def infodict(cont, title='', cols=4, son_cols=4,gap=4):
    have_son = False
    for i,j in cont.items():
        if type(j) == dict:
            if len(j) <= 10:
                have_son = False
            else:
                have_son = "son_dict"
        elif type(j) == list:
            have_son = "son_list"
        elif type(j) == set:
            have_son = "son_set"
        else:
            have_son = False
    logger.info("-" * 72)
    logger.info(r">>> {} (Type:{},Len:{})".format(
        title, str(type(cont))[8:-2], len(cont)))
    logger.info("-" * 72)
    if not have_son:
        holder = ["{} -> {}".format(i, cont[i]) for i in cont.keys()]
        logger.info(list_columns(holder, cols=cols,gap=gap))
    elif have_son == "son_dict":
        for i,j in cont.items():
            if j:
                infodict_son(j, title=title+"['"+i+"'-> ]", cols=son_cols)
    elif have_son == "son_list":
        # infolist_son(j, title=title+"["+str(i)+"]", cols=son_cols)
        for i,j in cont.items():
            if j:
                infolist(j, title=title+"["+str(i)+"]", cols=son_cols)
    elif have_son == "son_set":
        if max([len(j) for i,j in cont.items()])<27:
            for i,j in cont.items():
                if j:
                    logger.info("{}({}) -> {}".format(i,"%02d"%len(j),set(list(j)[:12])))
                    if len(j) > 12:
                        # logger.info("         {}".format(set(list(j)[12:24])))
                        logger.info("{}{}".format(' '*(len(str(i))+8),set(list(j)[12:24])))
                        if len(j) > 24:
                            # logger.info("         {}".format(set(list(j)[24:])))
                            logger.info("{}{}".format(' '*(len(str(i))+8),set(list(j)[24:])))
                    # logger.info("{} -> {}".format(i,j))
                    # info("{} -> {}".format(i,j),'',1)
        else:
            for i,j in cont.items():
                if j:
                    if len(j) == 0:
                        have_son = False
                        # continue
                    else:
                        infoset_son(j, title=title+"{"+str(i)+"}", cols=son_cols)
    else:
        for i,j in cont.items():
            info(j, title=title+":"+i, cols=1)
    logger.info("-" * 72)
    logger.info("\n")

def infoset(cont, title='', cols=1, son_cols=4,gap=4):
    logger.info("-" * 72)
    logger.info(r">>> {} (Type:{},Len:{})".format(
        title, str(type(cont))[8:-2], len(cont)))
    logger.info("-" * 72)
    logger.info(list_columns(cont, cols=cols,gap=gap))
    logger.info("-" * 72)
    logger.info("\n")

def infosimple(cont,title='',cols=1):
    logger.info("-" * 72)
    logger.info(r">>> {} (Type:{},Len:{})".format(
        title, str(type(cont))[8:-2], len(cont)))
    logger.info("-" * 72)
    cont = cont if type(cont) == str else ', '.join(cont)
    logger.info(cont)
    logger.info("-" * 72)
