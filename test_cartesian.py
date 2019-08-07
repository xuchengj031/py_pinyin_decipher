import itertools
from py_formatted_log import *
flat = lambda lst: sum(map(flat, lst), []) if isinstance(lst, list) else [lst]
solved_alpha = []
remain_alpha = list("zsbfkmpqtw")
# unsolved_symbol = list("ZUY")
# unsolved_symbol = list("SAZTUY")
# unsolved_symbol = list("DQ")
# unsolved_symbol = list("SAZTUYMXVR")
unsolved_symbol = list("SAZTUYMVDQ")
suspects_dict={
    'b':{'S', 'A'},
    'f':{'T', 'U', 'Z'},
    'k':{'Y', 'U'},
    'm':{'M', 'T', 'Z', 'S', 'A'},
    'p':{'U', 'Y', 'Z'},
    'q':{'M', 'A'},
    's':{'D', 'Q'},
    't':{'M', 'A', 'V'},
    'w':{'T', 'Z'},
    # 'x':{'X'},
    # 'y':{'R'}
    'z':{'D', 'Q'}
}

# lst = list("mdt")
lst = list("bfkmpqtwzs")
# lst = list("zsc")
lst = list(set(lst).intersection(set(remain_alpha)))
# lst = list("bfpmdt")
# occ = list("SAZTUY")
# list_prod_length = 1
list_to_prod = []
cartesian_prod = []
# tmp_unsolved_symbol=list(suspects_dict[k]) for k in lst
tmp_unsolved_symbol=list(set(flat([list(suspects_dict[k]) for k in lst])))
suspects_reverse_dict = dict(zip(tmp_unsolved_symbol, [[] for i in range(len(tmp_unsolved_symbol))]))
# tmp_progress_dict = {}

# for i in lst:
#     list_prod_length *= len(suspects_dict[i])

for i in tmp_unsolved_symbol:
    for j in lst:
        if i in suspects_dict[j]:
            suspects_reverse_dict[i].append(j)
for i in tmp_unsolved_symbol:
    list_to_prod.append(list(suspects_reverse_dict[i]))



infolist(list_to_prod,"list_to_prod",2)
infodict(suspects_reverse_dict,"suspects_reverse_dict",2)
for i in itertools.product(*list_to_prod):
    cartesian_prod.append(i)
cartesian_prod = list(filter(lambda x: len(set(x))== len(x), cartesian_prod))


infolist(cartesian_prod,"cartesian_prod",1)

tmp_couple_lst=[]
for i in range(len(cartesian_prod)):
    tmp_couples=[]
    for j in range(len(tmp_unsolved_symbol)):
        tmp_couples.append((tmp_unsolved_symbol[j],cartesian_prod[i][j]))
    tmp_couples.sort()
    infoset(tmp_couples,"tmp_couples",4)
    tmp_couple_lst.append(tmp_couples)
infolist(tmp_couple_lst,"tmp_couple_lst",2)
# for i in tmp_unsolved_symbol:
#     list_to_prod.append(list(suspects_reverse_dict[i]))
# infodict(suspects_reverse_dict,"suspects_reverse_dict",1)

tmp_progress_dict = {}
for i in tmp_couple_lst:
    tmp_progress_dict.update(dict(zip([tuple(i) for i in tmp_couple_lst],['' for i in range(len(tmp_couple_lst))])))


for i,j in tmp_progress_dict.items():
    tmp_progress_dict[i]="ni hao ya"
for i,j in tmp_progress_dict.items():
    info(j,"{}".format(i))
# info(tuple(tmp_couple_lst[0]))

# print(tmp_progress_dict,"tmp_progress_dict",2)
