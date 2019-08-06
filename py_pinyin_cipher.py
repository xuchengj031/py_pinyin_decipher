import re
import csv
import collections

import numpy as np

from py_formatted_log import *



def gen_word_list(msg):
    lst = msg.split() if type(msg) == str else msg
    word_max_len = max([len(i) for i in lst])
    word_list = [set() for i in range(word_max_len + 1)]
    for word in lst:
        for x in range(1, len(lst)):
            if len(word) == x:
                word_list[x].add(word)
    return [list(i) for i in word_list]


def stat_pos(msg, char_list):
    legal_pos_dict = dict(
        zip(char_list, [set() for x in range(len(char_list))]))
    legal_pinyin_list = gen_word_list(msg)
    for letter in char_list:
        for i in legal_pinyin_list:
            for j in i:
                if j.find(letter) != -1:
                    legal_pos_dict[letter].add((len(j), j.find(letter) + 1))
    return (legal_pos_dict, legal_pinyin_list)


def verifier(c_char, p_char, msg):
    '''检验提交的替换是否有效
    '''
    flag = True
    tmp_progress = msg.replace(c_char, p_char)
    cond1 = c_char.isupper()
    cond2 = p_char.islower()
    cond3 = c_char in unsolved_symbol
    cond4 = p_char in remain_alpha
    cond5 = c_char not in never_dict[p_char]
    cond6 = c_char in legal_dict[p_char]

    for word in tmp_progress.split():
        if word.find(p_char) == -1:
            continue
        else:
            for x in [i.start() for i in re.finditer(p_char, word)]:
                l = len(word)
                if p_char == "g" and x > 0:
                    if "n" in solved_alpha and word[x - 1] != "n":
                        return False
                if p_char == "n" and 0 < x < len(word):
                    if "g" in solved_alpha:
                        if not any([word.endswith("g"), word.endswith("n")]):
                            return False
                if p_char == "h" and x > 0:
                    if x != 1:
                        return False
                    if legal_dict["z"]:
                        if word[x - 1] not in list(legal_dict["z"]) + list("zcs"):
                            return False
    if p_char not in remain_alpha:
        info("{} -> {}: {} {} {} {} {} {}".format(c_char, p_char,
                                              cond1, cond2, cond3, cond4, cond5, cond6), "verifier")
    return all([cond1, cond2, cond3, cond4, cond5, cond6])


def updater(msg, dic):
    '''检查提交的嫌疑字典是否有唯一值，有则替换，
    并更新字典再次检查，直到字典里不再有唯一值
    '''
    progress = msg
    need_to_update = True

    def update_dict(dict_to_update):
        to_remove = []
        # 如果字典中某项有唯一值，就将它添加到couples
        for i, j in dict_to_update.items():
            if len(j) == 1:
                c_char = list(j)[0]
                couples.add((c_char, i))
                to_remove.append((i, j))
        # 以下两个循环，是将已确定配对字母的符号，从其它字母嫌疑单里去掉
        for i, j in mapping_dict.items():
            if j:
                to_remove.append((j, set(i)))
        for i in to_remove:
            for j, k in dict_to_update.items():
                if type(k) == set:
                    if j != i[0] and i[1].issubset(k):
                        dict_to_update[j] = dict_to_update[j] - i[1]
        # 所有嫌疑字典里都排除掉never字典里的名单
        for i in remain_alpha:
            legal_dict[i] = legal_dict[i] - never_dict[i]
            for j in [suspects_dict, presumably_dict,probably_dict, possibly_dict]:
                if j:
                    j[i] = j[i].intersection(legal_dict[i])

        # 返回该字典里剩余具有唯一值的键的个数
        engaged = [k for k in dict_to_update.values() if len(k) == 1]
        info("engaged unique: {}\t{}".format(
            engaged, len(engaged)), 'update_dict: {}'.format(''))

        return len([k for k in dict_to_update.values() if len(k) == 1])

    def check_unique_value(dict_to_check):
        all_solved = set([i for i in dict_to_check.keys() if len(
            dict_to_check[i]) == 1]).issubset(set(solved_alpha))
        info(all_solved, 'check_unique_value: {}'.format(''))
        return all_solved

    def committer(c_char, p_char, msg):
        '''替换 (符号, 字母) 对
        '''
        c_char = list(c_char)[0] if type(c_char) == set else str(c_char)
        progress = msg
        progress = progress.replace(c_char, p_char)
        info(progress[:300], 'committer ({} -> {})'.format(c_char, p_char))
        unsolved_symbol.remove(c_char)
        remain_alpha.remove(p_char)
        solved_alpha.append(p_char)
        mapping_dict[c_char] = p_char
        return progress

    def update_couples(msg):
        progress = msg

        for i in list(couples):
            if i[1] in remain_alpha:
                if verifier(i[0], i[1], msg):
                    progress = committer(i[0], i[1], progress)
                    info((i[0], i[1]), 'update_couples: {}'.format(''))

        return progress

    need_to_update = update_dict(dic)
    # 只要字典里还有唯一值，就一直替换、更新字典、再检查有无唯一值……
    while need_to_update:
        if not check_unique_value(dic):
            progress = update_couples(progress)
            need_to_update = update_dict(dic)
        # 这个else不可去掉，否则会死循环
        else:
            break

    progress = update_couples(progress)
    return progress


def letter_pos_analyse(fname):

    with open('data/all_legal_pinyin.txt', 'r', encoding="utf-8") as fp:
        all_legal_pinyin = fp.read()
    # infolist(all_legal_pinyin.split(), "all_legal_pinyin", 7)

    # 统计正确拼音中每个字母的位置
    legal_pos_dict, legal_pinyin_list = stat_pos(all_legal_pinyin, lower_case)
    # 统密文中每个字母的位置
    occ_pos_dict, occ_pinyin_list = stat_pos(article, upper_case[:total_symbos])

    # 如果密文中符号的位置的集合，正好是某字母正确位置的子集，那它可能是这个字母；
    # 反之如果不在该字母位置集合的范围内，它就绝对不会是这个字母
    for i, j in occ_pos_dict.items():
        for k, l in legal_pos_dict.items():
            if j.issubset(l):
                legal_dict[k] = legal_dict[k].union(i)
                legal_dict_copy[k] = legal_dict[k].union(i)
            else:
                never_dict[k] = never_dict[k].union(i)

    infodict(legal_dict, "legal_dict:begining", 1, 14)
    infodict(never_dict, "never_dict:begining", 1, 14)

    has_single_pos = [i for i in legal_pos_dict.keys() if max(
        [j[1] for j in legal_pos_dict[i]]) < 2]
    has_multi_pos = list(set(lower_case) - set(has_single_pos))
    has_single_pos_set = set()
    has_multi_pos_set = set()
    for i in occ_pos_dict.keys():
        # 考虑到密文符号不足26个的情况
        if occ_pos_dict.get(i):
            if max([j[1] for j in occ_pos_dict[i]]) < 2:
                has_single_pos_set.add(i)
            else:
                has_multi_pos_set.add(i)

    # 位置多变的符号一定不会是位置单一的字母
    for i in has_multi_pos:
        legal_dict[i] = legal_dict[i] - has_single_pos_set
        if total_chars > 400:
            never_dict[i] = never_dict[i].union(has_single_pos_set)
    for i in has_single_pos:
        legal_dict[i] = legal_dict[i] - has_multi_pos_set
        if total_chars > 400:
            never_dict[i] = never_dict[i].union(has_multi_pos_set)

    infolist(legal_pinyin_list, "legal_pinyin_list", 1, 8)
    infolist(occ_pinyin_list, "occ_pinyin_list", 1, 8)
    infodict(legal_pos_dict, "legal_pos_dict", 1, 7)
    infodict(occ_pos_dict, "occ_pos_dict", 1, 7)
    infosimple(has_single_pos, "has_single_pos", 20)
    infosimple(has_single_pos_set, "has_single_pos_set", 20)
    infosimple(has_multi_pos, "has_multi_pos", 20)
    infosimple(has_multi_pos_set, "has_multi_pos_set", 20)
    infodict(never_dict, "never_dict", 1, 14)

    def gen_reverse_dict(pos_dict):
        pos_set = set()
        for i, j in pos_dict.items():
            for k in j:
                pos_set.add(k)
        pos_reverse_dict = dict(
            zip(list(pos_set), [set() for x in range(len(pos_set))]))
        for i in list(pos_set):
            for j, k in pos_dict.items():
                if i in k:
                    pos_reverse_dict[i].add(j)
        pos_list = list(pos_set)
        pos_list = sorted(pos_list, key=lambda tup: (tup[0],tup[1]))
        tmp = sorted(pos_reverse_dict.items(), key=lambda tup: (tup[0],tup[1]))
        pos_reverse_dict = dict(zip([i[0] for i in tmp],[i[1] for i in tmp]))
        return [pos_list, pos_reverse_dict]

    # 生成反向的位置表，即以位置为键
    legal_pos_list, legal_pos_reverse_dict = gen_reverse_dict(legal_pos_dict)
    occ_pos_list, occ_pos_reverse_dict = gen_reverse_dict(occ_pos_dict)

    lucky_dict = dict(zip(lower_case, [set() for i in range(26)]))
    reverse_lucky_dict = dict(zip(upper_case[:total_symbos], [set() for i in range(26)]))
    # 这个lucky_dict很鸡肋，基本上只有出现6字拼音的时候才能才有用
    for i, j in legal_pos_reverse_dict.items():
        if j:
            if len(j) == 1 and occ_pos_reverse_dict.get(i):
                lucky_dict[list(j)[0]] = lucky_dict[list(j)[0]].union(occ_pos_reverse_dict[i])
    # 这个reverse_lucky_dict没啥用
    for i, j in occ_pos_reverse_dict.items():
        if j:
            if len(j) == 1:
                reverse_lucky_dict[list(j)[0]] = reverse_lucky_dict[
                    list(j)[0]].union(legal_pos_reverse_dict[i])

    infodict(legal_pos_reverse_dict, "legal_pos_reverse_dict", 1, 14)
    infodict(occ_pos_reverse_dict, "occ_pos_reverse_dict", 1)
    infodict(occ_pos_reverse_dict, "occ_pos_reverse_dict", 1, 14)
    infodict(lucky_dict, "lucky_dict", 1, 14)
    infodict(reverse_lucky_dict, "reverse_lucky_dict", 1, 14)

    progress = updater(article, lucky_dict)
    progress = updater(progress, legal_dict)

    infodict(legal_dict, "legal_dict:letter_pos_analyse", 1, 14)
    infoset(couples, "couples:letter_pos_analyse", 5)
    infoset(unsolved_symbol, "unsolved_symbol:letter_pos_analyse", 14)
    infoset(remain_alpha, "remain_alpha:letter_pos_analyse", 14)
    infoset(solved_alpha, "solved_alpha:letter_pos_analyse", 14)
    infodict(mapping_dict, "mapping_dict:letter_pos_analyse", 6)
    info(progress, "progress:letter_pos_analyse", 1)

    return [progress,legal_pinyin_list]

def letter_freq_analyse(msg):
    '''
    '''
    # 1.获取字母频率预期值
    # 按频率预期值排序的字母列表
    exp_letter_seq = []
    # 字母频率的预期频率
    exp_letter_freq = []
    with open("data/pinyin_letter_freq.csv", "r", encoding="utf-8") as fp:
        f_csv = csv.reader(fp)
        for row in f_csv:
            exp_letter_seq.append(row[0])
            exp_letter_freq.append(round(float(row[1]),4))
    exp_freq_dict = dict(zip(exp_letter_seq, exp_letter_freq))

    infodict(exp_freq_dict,"exp_freq_dict", 4)

    # 2.获取密文字符频率
    msg_without_space = msg.replace(" ","")
    # 统计密文中各符号出现的次数
    alpha_dict = collections.Counter(msg_without_space)
    # 计算密文中去除空格后的总字符数
    amount = len(msg_without_space)
    # 按频率排序后的列表
    occ_rank = sorted(alpha_dict.items(), key=lambda x: x[1], reverse=True)

    infolist(occ_rank,"occ_rank: amount({})".format(amount), 5)

    occ_alphabet = [i for i, j in occ_rank]
    occ_alpha = [i for i, j in occ_rank]
    occ_freq = [round(j / amount * 100,4) for i, j in occ_rank]
    occ_freq_dict = dict(zip(occ_alphabet, occ_freq))

    infodict(occ_freq_dict,"occ_freq_dict", 4)

    # 3.交叉对比以上两个频率表
    failed = 0
    vibration = 0
    def check_freq(cur_symbol, letter, vibration=0):
        if vibration == 0:
            vibration = 1
        return abs(float(exp_freq_dict[letter]) - occ_freq_dict[cur_symbol]) < vibration
    def prober(letter, msg, vibration, failed):
        '''
        TODO：还需考虑到密文中字母不全的情况
        '''
        # symbos_count=len(set(msg))
        # adapter = symbos_count / 26
        adapter = total_symbos / 26
        # if total_chars < 400:
        #     adapter *= (total_chars / 400)
        info("symbos: {}\tadapter: ({})".format(total_symbos,adapter),"total_symbos & adapter {}".format(letter))
        if letter in remain_alpha:
            x = round(exp_letter_seq.index(letter) * adapter)
            # info("curr letter '{}' rank {}: vibration={},failed={}".format(letter,str(x),str(vibration),str(failed)))

            idxs = set()
            suspects = []
            if not vibration:
                if x < 10*adapter:
                    if letter in list("nuhc"):
                        vibration = 2
                    elif letter in list("aegz"):
                        vibration = 1.7
                    else:
                        vibration = 1.5
                else:
                    if letter in list("im"):
                        vibration = 1.5
                    elif x > 21*adapter:
                        vibration = 0.5
                    else:
                        vibration = 1

            if letter in list("umwzcsrq"):
                inc = round(5/adapter)
                dec = round(-5/adapter)
            elif letter in list("tylgd"):
                inc = round(4/adapter)
                dec = round(-4/adapter)
            elif letter in list("inaveh"):
                inc = round(2/adapter)
                dec = round(-2/adapter)
            else:
                inc = round(3/adapter)
                dec = round(-3/adapter)
            info("{}: [{}/{}]".format(letter,inc,dec),"{}: inc/dec".format(letter))

            for i in range(dec, inc):
                if x + i < total_symbos:
                    idxs.add((x + i) if x + i > 0 else 0)
            infoset(idxs,"{}[{}]: idxs".format(letter,x),13,gap=2)
            # info("curr letter '{}' updated: vibration={},idxs=[{}]".format(letter,str(vibration),','.join([str(i) for i in idxs])))
            for idx in idxs:
                # 考虑到密文符号不足26个的情况
                if idx < len(occ_alphabet):
                    cur_symbol = occ_alphabet[idx]
                    tmp_msg = msg.replace(cur_symbol, letter)
                    if verifier(cur_symbol,letter,tmp_msg) & check_freq(cur_symbol, letter, vibration/adapter):
                        suspects.append(cur_symbol)
                        info("suspects append '{}' for '{}'".format(cur_symbol,letter))
            if len(set(suspects)) == 1:
                suspects_dict[letter] = set(suspects[0])
                failed = 0
            elif len(set(suspects)) == 0:
                failed = 1

            else:
                failed = 0
                for i in suspects:
                    suspects_dict[letter].add(i)
            # info("curr letter '{}' updated: vibration={},failed={}".format(letter,str(vibration),str(failed)))
        return (failed, vibration)

    def guess_by_freq(letters, msg, vibration):

        for letter in list(letters):
            x = prober(letter, msg, vibration, 0)
            failed = x[0]
            vibration = x[1]
            if failed == 1:
                for i in np.arange(vibration, vibration + 1.5, 0.1):
                    x = prober(letter, msg, i, 1)
                    failed = x[0]
                    vibration = x[1]
                    if failed == 0:
                        break

    # guess_by_freq("ivaeounhg", msg, vibration)
    guess_by_freq(remain_alpha, msg_without_space, vibration)
    for i in lower_case:
        if type(suspects_dict[i]) == set:
            suspects_dict[i] = suspects_dict[i].intersection(legal_dict[i])
    infodict(suspects_dict, "suspects_dict:letter_freq_analyse:begining", 1, 14)

    for i in remain_alpha:
        if not suspects_dict[i].issubset(legal_dict[i]):
            info("warning:sth rong!", "warning:letter_freq_analyse {}".format(i), 1)


    infodict(legal_dict, "legal_dict:b4_letter_freq_analyse", 1, 14)
    infodict(suspects_dict, "suspects_dict:b4_letter_freq_analyse", 1, 14)
    progress = msg
    progress = updater(progress, {k: suspects_dict[k] for k in suspects_dict.keys() if k in list("aoeiuv")})
    infodict({k: suspects_dict[k] for k in suspects_dict.keys() if k in list("aoeiuv")}, "legal_dict:vowel", 1, 14)
    progress = updater(progress, legal_dict)
    # progress = updater(progress, suspects_dict)
    infodict(legal_dict, "legal_dict:letter_freq_analyse", 1, 14)
    infodict(suspects_dict, "suspects_dict:letter_freq_analyse", 1, 14)
    infoset(couples, "couples:letter_freq_analyse", 5)
    infoset(unsolved_symbol, "unsolved_symbol:letter_freq_analyse", 14)
    infoset(remain_alpha, "remain_alpha:letter_freq_analyse", 14)
    infoset(solved_alpha, "solved_alpha:letter_freq_analyse", 14)
    infodict(mapping_dict, "mapping_dict:letter_freq_analyse", 7)
    # info(progress, "progress:letter_freq_analyse", 1)

    return progress

def spell_analyse(msg):
    '''
    '''
    # 所有韵母的列表，按字符数分
    pinyin_finals = "a,o,e,i,u,v,ai,ao,an,ou,ei,en,er,ia,ie,in,iu,ua,ue,un,uo,ui,ve,ang,ong,eng,ing,iao,ian,uai,uan,iang,iong,uang".split(
    ',')
    occ_final = dict(zip(pinyin_finals, [set()
                                         for i in range(len(pinyin_finals))]))
    exp_final = dict(zip(pinyin_finals, [set()
                                         for i in range(len(pinyin_finals))]))
    finals = gen_word_list(pinyin_finals)
    infolist(finals,"finals", 1,12)
    infolist(legal_pinyin_list,"legal_pinyin_list:spell_analyse",1,8)

    initial_list = list("nghrbpmfdtlkjqxzcsyw")
    exp_initial_dict = dict(zip(initial_list,[set() for i in range(len(initial_list))]))
    legal_initial_dict = dict(zip(initial_list,[set() for i in range(len(initial_list))]))
    legal_initial_reverse_dict = dict(zip(pinyin_finals,[set() for i in range(len(pinyin_finals))]))
    never_dict1 = dict(zip(lower_case, [set() for i in range(26)]))
    never_dict2 = dict(zip(lower_case, [set() for i in range(26)]))
    flat=lambda lst: sum(map(flat,lst),[]) if isinstance(lst,list) else [lst]
    for letter in initial_list:
        if letter in remain_alpha:
            for word in flat(legal_pinyin_list[2:]):
                if word.startswith(letter):
                    if word[1] != "h":
                        legal_final=word.lstrip(letter)
                    else:
                        legal_final=word.lstrip(letter).lstrip("h")
                    exp_initial_dict[letter].add(legal_final)
                    # info(legal_final,"legal_final")
    pat=r'^[nghrbpmfdtlkjqxzcsyw]h?'
    for final in pinyin_finals:
        for word in flat(legal_pinyin_list[2:]):
            # if word.endswith(final) and word[0] not in list("aoe"):
            # print(re.sub(pat,'',word),word,final)
            if re.sub(pat,'',word) == final:
                if word[0] not in list("aoeiuv"):
                    legal_initial_reverse_dict.get(final).add(word[0])

    infodict(exp_initial_dict,"exp_initial_dict", 1)

    tmp_words = msg.split()
    infolist(tmp_words, "tmp_words", 7)
    for f in pinyin_finals:
        for word in tmp_words:
            if word.endswith(f):
                if word[:word.find(f)].rstrip("h").isupper():
                    occ_final[f].add(word[:word.find(f)].strip('h'))
        for o in flat(legal_pinyin_list):
            if o.endswith(f):
                gross_init = re.sub('[aoeiuv]', '', o[:o.find(f)])
                pure_init = gross_init.strip().rstrip('h')
                exp_final[f].add(pure_init)

    zcs_set = set()
    for word in [i for i in tmp_words if len(i) == 2 or len(i) == 3]:
        if word[1] == "v" and word[0].isupper():
            for j in list("nl"):
                if "n" in remain_alpha:
                    nl_set.add(word[0])
                else:
                    suspects_dict["l"] = set(word[0])
                    couples.add((word[0], "l"))
        if word[0] == "e" and len(word) == 2 and word[1].isupper():
            for j in list("rni"):
                if "n" in remain_alpha or "i" in remain_alpha:
                    rni_set.add(word[1])
                else:
                    suspects_dict["r"] = set(word[1])
                    couples.add((word[1], "r"))
    for word in [i for i in tmp_words if len(i) > 2]:
        if "h" in solved_alpha:
            if word.find("h") == 1:
                zcs_set.add(word[word.find("h")-1])
    for i in suspects_dict:
        if type(suspects_dict[i]) == set:
            if i in list("zcs"):
                suspects_dict[i] = suspects_dict[i].intersection(zcs_set)
    for i in legal_dict:
        if type(legal_dict[i]) == set:
            if i in list("zcs"):
                legal_dict[i] = legal_dict[i].intersection(zcs_set)
    infodict(legal_dict, "legal_dict:b4_screen_zcs", 1, 14)
    infodict(suspects_dict, "suspects_dict:b4_screen_zcs", 1, 14)
    for i in "zcs":
        for j in zcs_set:
            for word in [k for k in tmp_words if len(k) > 2]:
                if word.replace(j,i) in ["cei","chei","sei","shong"]:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen zcs")
                    if j in legal_dict[i]:
                        legal_dict[i].remove(j)
                    if j in suspects_dict[i]:
                        suspects_dict[i].remove(j)
    to_remove_from_dict = set()
    for i in "bpmf":
        for j in legal_dict[i]:
            for word in [k for k in tmp_words if 5 > len(k) > 1]:
                if word.replace(j,i) in ['me']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen bpmf")
                    if set("dtnlgkhzcsr").issubset(solved_alpha):
                        legal_dict['m']=set(j)
                    # legal_dict['m']=set(j)
                    to_remove_from_dict.add(("b",j))
                    to_remove_from_dict.add(("p",j))
                    to_remove_from_dict.add(("f",j))
                if word.replace(j,i) in ['miu']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen bpmf")
                    if set("dnljqx").issubset(solved_alpha):
                        legal_dict['m']=set(j)
                    # legal_dict['m']=set(j)
                    to_remove_from_dict.add(("b",j))
                    to_remove_from_dict.add(("p",j))
                    to_remove_from_dict.add(("f",j))
                if word.replace(j,i) in ['fai','fao','fe','fi','fiao','fie','fiu','fian','fin','fing','be','bou','biu','pe','piu']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen bpmf")
                    to_remove_from_dict.add((i,j))

    for i in "dtl":
        for j in legal_dict[i]:
            for word in [k for k in tmp_words if 6 > len(k) > 1]:
                if word.replace(j,i) in ['lo']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen dtl")
                    if set("bpmf").issubset(solved_alpha):
                        legal_dict['l']=set(j)
                    # legal_dict['l']=set(j)
                    to_remove_from_dict.add(("d",j))
                    to_remove_from_dict.add(("t",j))
                # if word.replace(j,i) in ['lia']:
                #     info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen dtl")
                #     if set("jqx").issubset(solved_alpha):
                #         legal_dict['l']=set(j)
                #     to_remove_from_dict.add(("d",j))
                #     to_remove_from_dict.add(("t",j))
                if word.replace(j,i) in ['lin']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen dtl")
                    if set("bpmjqx").issubset(solved_alpha):
                        legal_dict['l']=set(j)
                    # legal_dict['l']=set(j)
                    to_remove_from_dict.add(("d",j))
                    to_remove_from_dict.add(("t",j))
                if word.replace(j,i) in ['liang']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen dtl")
                    if set("jqx").issubset(solved_alpha):
                        legal_dict['l']=set(j)
                    # legal_dict['l']=set(j)
                    to_remove_from_dict.add(("d",j))
                    to_remove_from_dict.add(("t",j))
                if word.replace(j,i) in ['den']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen dtl")
                    if set("bpmfgkhzcsr").issubset(solved_alpha):
                        legal_dict['d']=set(j)
                    # legal_dict['d']=set(j)
                    to_remove_from_dict.add(("l",j))
                    to_remove_from_dict.add(("t",j))
                if word.replace(j,i) in ['tei','tiu','tin','din','tia','diang','tiang','lui']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen dtl")
                    to_remove_from_dict.add((i,j))
    for i in to_remove_from_dict:
        if i[1] in legal_dict.get(i[0]):
            legal_dict[i[0]].remove(i[1])
        if i[1] in suspects_dict.get(i[0]):
            suspects_dict[i[0]].remove(i[1])
    infoset(zcs_set,"zcs_set:spell_analyse",12)
    infoset(to_remove_from_dict,"to_remove_from_dict:spell_analyse",12)
    infodict(legal_dict, "legal_dict:after_screen_zcs", 1, 14)
    infodict(suspects_dict, "suspects_dict:after_screen_zcs", 1, 14)

    match_list = [[i for i in occ_final.keys()], [a for a in occ_final.values()], [
        o for o in exp_final.values()]]
    occ_initial_dict=dict(zip([i for i in occ_final.keys()], [a for a in occ_final.values()]))

    for i,j in exp_initial_dict.items():
        if j:
            for k in j:
                for l,m in occ_initial_dict.items():
                    legal_initial_dict[i] = legal_initial_dict[i].union(occ_initial_dict[l])

    infodict(occ_initial_dict,"occ_initial_dict", 1)
    infodict(legal_initial_dict,"legal_initial_dict", 1)
    infodict(legal_initial_reverse_dict,"legal_initial_reverse_dict", 1)

    lst_tmp = []
    for i in range(len(match_list[2])):
        if match_list[1][i]:
            lst_tmp.append([match_list[1][i], match_list[2][i]])

    infolist(match_list,"match_list",1,1)
    infolist(lst_tmp,"lst_tmp",1,1)

    for i in remain_alpha:
        for j in legal_dict[i]:
            for word in tmp_words:
                if word.find(j) >= 0:
                    if not word.replace(j, i) in legal_pinyin_list[len(word)]:
                        if i not in "zcs":
                            never_dict1[i].add(j)

    infodict(legal_dict, "legal_dict:b4_never_dict", 1, 14)
    infodict(suspects_dict, "suspects_dict:b4_never_dict", 1, 14)
    infodict(never_dict, "never_dict:b4_never_dict", 1, 14)
    infodict(never_dict1, "never_dict1:b4_never_dict", 1, 14)
    infodict(never_dict2, "never_dict2:b4_never_dict", 1, 14)

    for i in remain_alpha:
        never_dict[i]=never_dict[i].union(never_dict1[i])
        suspects_dict[i]=suspects_dict[i]-never_dict[i]
        legal_dict[i]=legal_dict[i]-never_dict[i]
        presumably_dict[i]=presumably_dict[i]-never_dict[i]
        probably_dict[i]=probably_dict[i]-never_dict[i]
        possibly_dict[i]=possibly_dict[i]-never_dict[i]



    infoset(zcs_set, "zcs_set:b4_spell_analyse", 1, 14)
    infodict(legal_dict, "legal_dict:b4_spell_analyse", 1, 14)
    infodict(suspects_dict, "suspects_dict:b4_spell_analyse", 1, 14)
    progress = msg
    progress = updater(progress, legal_dict)
    # progress = updater(progress, {k: suspects_dict[k] for k in suspects_dict.keys() if k in remain_alpha})
    # progress = updater(progress, legal_dict)

    infodict(legal_dict, "legal_dict:spell_analyse", 1, 14)
    infodict(suspects_dict, "suspects_dict:spell_analyse", 1, 14)
    infoset(couples, "couples:spell_analyse", 5)
    infoset(unsolved_symbol, "unsolved_symbol:spell_analyse", 14)
    infoset(remain_alpha, "remain_alpha:spell_analyse", 14)
    infoset(solved_alpha, "solved_alpha:spell_analyse", 14)
    infodict(mapping_dict, "mapping_dict:spell_analyse", 7)
    info(progress, "progress:spell_analyse", 1)

    return progress


# fname="留言2_op_已加密_转写"
# fname="来自白犀的问候_op_已加密_转写"
# fname="因为门德尔松_op_已加密_转写"
# fname="随想_op_已加密_转写"
fname="留言1-1_op_已加密_转写"
with open("src/" + fname + ".txt", "r", encoding="utf-8") as fp:
    article = fp.read()
    article_without_space = article.replace(" ", "")
    total_chars = len(article_without_space)
    total_words = len(article.split())
    occ_symbos = list(set(article_without_space))
    total_symbos = len(occ_symbos)
    info("chars:{}, words:{}, symbos:{}".format(total_chars,total_words,total_symbos))
    infolist(occ_symbos,"occ_symbos",14)
    lower_case = [chr(i) for i in range(97, 123)]
    upper_case = [chr(i) for i in range(65, 91)]
    # 待替换的字母对 (大写字母, 小写字母)
    couples = set()
    # 明码密码对照表
    mapping_dict = dict(zip(upper_case[:total_symbos], ['' for i in range(total_symbos)]))
    # 未破译的密文符号 (大写)
    unsolved_symbol = upper_case[:total_symbos]
    # 未破译的字母 (小写)
    remain_alpha = [chr(i) for i in range(97, 123)]
    # 已破译的字母 (小写)
    solved_alpha = []
    # 由字母位置得出，是密文的最大嫌疑范围，逐步缩窄
    legal_dict = dict(zip(lower_case, [set() for i in range(26)]))
    # 留个备份，以免误操作
    legal_dict_copy = dict(zip(lower_case, [set() for i in range(26)]))
    # 绝对排除嫌疑
    never_dict = dict(zip(lower_case, [set() for i in range(26)]))
    # 由字母频率分析得来的嫌疑名单
    suspects_dict = dict(zip(lower_case, [set() for i in range(26)]))
    # 由词频分析得来的嫌疑名单
    presumably_dict = dict(zip(lower_case, [set() for i in range(26)]))
    probably_dict = dict(zip(lower_case, [set() for i in range(26)]))
    possibly_dict = dict(zip(lower_case, [set() for i in range(26)]))

progress,legal_pinyin_list=letter_pos_analyse(fname)
progress=letter_freq_analyse(progress)
progress=spell_analyse(progress)
progress=spell_analyse(progress)
progress = updater(progress, legal_dict)
# progress = updater(progress, {k: suspects_dict[k] for k in suspects_dict.keys() if k in remain_alpha})
progress = updater(progress, legal_dict)
# progress = updater(progress, suspects_dict)

infodict({k: legal_dict[k] for k in legal_dict.keys() if k in remain_alpha}, "legal_dict:now", 1, 14)
infodict({k: suspects_dict[k] for k in suspects_dict.keys() if k in remain_alpha}, "suspects_dict:now", 1, 14)
infoset(couples, "couples:now", 5)
infoset(unsolved_symbol, "unsolved_symbol:now", 14)
infoset(remain_alpha, "remain_alpha:now", 14)
infoset(solved_alpha, "solved_alpha:now", 14)
infodict(mapping_dict, "mapping_dict:now", 7)
info(progress, "progress:now", 1)
