import re
import csv
import collections
import itertools

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

flat = lambda lst: sum(map(flat, lst), []) if isinstance(lst, list) else [lst]
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
            for j in [suspects_dict, presumably_dict, probably_dict, possibly_dict]:
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
    occ_pos_dict, occ_pinyin_list = stat_pos(
        article, upper_case[:total_symbos])

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
        pos_list = sorted(pos_list, key=lambda tup: (tup[0], tup[1]))
        tmp = sorted(pos_reverse_dict.items(),
                     key=lambda tup: (tup[0], tup[1]))
        pos_reverse_dict = dict(zip([i[0] for i in tmp], [i[1] for i in tmp]))
        return [pos_list, pos_reverse_dict]

    # 生成反向的位置表，即以位置为键
    legal_pos_list, legal_pos_reverse_dict = gen_reverse_dict(legal_pos_dict)
    occ_pos_list, occ_pos_reverse_dict = gen_reverse_dict(occ_pos_dict)

    lucky_dict = dict(zip(lower_case, [set() for i in range(26)]))
    reverse_lucky_dict = dict(
        zip(upper_case[:total_symbos], [set() for i in range(26)]))
    # 这个lucky_dict很鸡肋，基本上只有出现6字拼音的时候才能才有用
    for i, j in legal_pos_reverse_dict.items():
        if j:
            if len(j) == 1 and occ_pos_reverse_dict.get(i):
                lucky_dict[list(j)[0]] = lucky_dict[list(
                    j)[0]].union(occ_pos_reverse_dict[i])
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

    return [progress, legal_pinyin_list]


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
            exp_letter_freq.append(round(float(row[1]), 4))
    exp_freq_dict = dict(zip(exp_letter_seq, exp_letter_freq))

    infodict(exp_freq_dict, "exp_freq_dict", 4)

    # 2.获取密文字符频率
    msg_without_space = msg.replace(" ", "")
    # 统计密文中各符号出现的次数
    alpha_dict = collections.Counter(msg_without_space)
    # 计算密文中去除空格后的总字符数
    amount = len(msg_without_space)
    # 按频率排序后的列表
    occ_rank = sorted(alpha_dict.items(), key=lambda x: x[1], reverse=True)

    infolist(occ_rank, "occ_rank: amount({})".format(amount), 5)

    occ_alphabet = [i for i, j in occ_rank]
    occ_alpha = [i for i, j in occ_rank]
    occ_freq = [round(j / amount * 100, 4) for i, j in occ_rank]
    occ_freq_dict = dict(zip(occ_alphabet, occ_freq))

    infodict(occ_freq_dict, "occ_freq_dict", 4)

    # 3.交叉对比以上两个频率表
    failed = 0
    vibration = 0

    def check_freq(cur_symbol, letter, vibration=0):
        if vibration == 0:
            vibration = 1
        return abs(float(exp_freq_dict[letter]) - occ_freq_dict[cur_symbol]) < vibration

    def prober(letter, msg, vibration, failed):
        '''
        TODO：还需对密文中字母不全的情况做更好应对
        '''
        adapter = total_symbos / 26
        # if total_chars < 400:
        #     adapter *= (total_chars / 400)
        info("symbos: {}\tadapter: ({})".format(total_symbos, adapter),
             "total_symbos & adapter {}".format(letter))
        if letter in remain_alpha:
            x = round(exp_letter_seq.index(letter) * adapter)
            # info("curr letter '{}' rank {}: vibration={},failed={}".format(letter,str(x),str(vibration),str(failed)))

            idxs = set()
            suspects = []
            if not vibration:
                if x < 10 * adapter:
                    if letter in list("nuhc"):
                        vibration = 2
                    elif letter in list("aegz"):
                        vibration = 1.7
                    else:
                        vibration = 1.5
                else:
                    if letter in list("im"):
                        vibration = 1.5
                    elif x > 21 * adapter:
                        vibration = 0.5
                    else:
                        vibration = 1

            if letter in list("umwzcsrq"):
                inc = round(5 / adapter)
                dec = round(-5 / adapter)
            elif letter in list("tylgd"):
                inc = round(4 / adapter)
                dec = round(-4 / adapter)
            elif letter in list("inaveh"):
                inc = round(2 / adapter)
                dec = round(-2 / adapter)
            else:
                inc = round(3 / adapter)
                dec = round(-3 / adapter)
            info("{}: [{}/{}]".format(letter, inc, dec),
                 "{}: inc/dec".format(letter))

            for i in range(dec, inc):
                if x + i < total_symbos:
                    idxs.add((x + i) if x + i > 0 else 0)
            infoset(idxs, "{}[{}]: idxs".format(letter, x), 13, gap=2)
            # info("curr letter '{}' updated: vibration={},idxs=[{}]".format(letter,str(vibration),','.join([str(i) for i in idxs])))
            for idx in idxs:
                # 考虑到密文符号不足26个的情况
                if idx < len(occ_alphabet):
                    cur_symbol = occ_alphabet[idx]
                    tmp_msg = msg.replace(cur_symbol, letter)
                    if verifier(cur_symbol, letter, tmp_msg) & check_freq(cur_symbol, letter, vibration / adapter):
                        suspects.append(cur_symbol)
                        info("suspects append '{}' for '{}'".format(
                            cur_symbol, letter))
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
    progress = updater(progress, {k: suspects_dict[
                       k] for k in suspects_dict.keys() if k in list("aoeiuv")})
    infodict({k: suspects_dict[k] for k in suspects_dict.keys(
    ) if k in list("aoeiuv")}, "legal_dict:vowel", 1, 14)
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
    infolist(finals, "finals", 1, 12)
    infolist(legal_pinyin_list, "legal_pinyin_list:spell_analyse", 1, 8)

    initial_list = list("nghrbpmfdtlkjqxzcsyw")
    exp_initial_dict = dict(
        zip(initial_list, [set() for i in range(len(initial_list))]))
    legal_initial_dict = dict(
        zip(initial_list, [set() for i in range(len(initial_list))]))
    legal_initial_reverse_dict = dict(
        zip(pinyin_finals, [set() for i in range(len(pinyin_finals))]))
    never_dict1 = dict(zip(lower_case, [set() for i in range(26)]))
    never_dict2 = dict(zip(lower_case, [set() for i in range(26)]))
    flat = lambda lst: sum(map(flat, lst), []) if isinstance(lst, list) else [lst]
    for letter in initial_list:
        if letter in remain_alpha:
            for word in flat(legal_pinyin_list[2:]):
                if word.startswith(letter):
                    if word[1] != "h":
                        legal_final = word.lstrip(letter)
                    else:
                        legal_final = word.lstrip(letter).lstrip("h")
                    exp_initial_dict[letter].add(legal_final)
                    # info(legal_final,"legal_final")
    pat = r'^[nghrbpmfdtlkjqxzcsyw]h?'
    for final in pinyin_finals:
        for word in flat(legal_pinyin_list[2:]):
            if re.sub(pat, '', word) == final:
                if word[0] not in list("aoeiuv"):
                    legal_initial_reverse_dict.get(final).add(word[0])

    infodict(exp_initial_dict, "exp_initial_dict", 1)

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
                zcs_set.add(word[word.find("h") - 1])
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
                if word.replace(j, i) in ["cei", "chei", "sei", "shong"]:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen zcs")
                    if j in legal_dict[i]:
                        legal_dict[i].remove(j)
                    if j in suspects_dict[i]:
                        suspects_dict[i].remove(j)
    to_remove_from_dict = set()
    for i in "bpmf":
        for j in legal_dict[i]:
            for word in [k for k in tmp_words if 5 > len(k) > 1]:
                if word.replace(j, i) in ['me']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen bpmf")
                    if set("dtnlgkhzcsr").issubset(solved_alpha):
                        legal_dict['m'] = set(j)
                    # legal_dict['m']=set(j)
                    to_remove_from_dict.add(("b", j))
                    to_remove_from_dict.add(("p", j))
                    to_remove_from_dict.add(("f", j))
                if word.replace(j, i) in ['miu']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen bpmf")
                    if set("dnljqx").issubset(solved_alpha):
                        legal_dict['m'] = set(j)
                    # legal_dict['m']=set(j)
                    to_remove_from_dict.add(("b", j))
                    to_remove_from_dict.add(("p", j))
                    to_remove_from_dict.add(("f", j))
                if word.replace(j, i) in ['fai', 'fao', 'fe', 'fi', 'fiao', 'fie', 'fiu', 'fian', 'fin', 'fing', 'be', 'bou', 'biu', 'pe', 'piu']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen bpmf")
                    to_remove_from_dict.add((i, j))

    for i in "dtl":
        for j in legal_dict[i]:
            for word in [k for k in tmp_words if 6 > len(k) > 1]:
                if word.replace(j, i) in ['lo']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen dtl")
                    if set("bpmf").issubset(solved_alpha):
                        legal_dict['l'] = set(j)
                    # legal_dict['l']=set(j)
                    to_remove_from_dict.add(("d", j))
                    to_remove_from_dict.add(("t", j))
                # if word.replace(j,i) in ['lia']:
                #     info("remove {} from legal_dict[{}] cos {}/{}".format(j,i,word,word.replace(j,i)),"screen dtl")
                #     if set("jqx").issubset(solved_alpha):
                #         legal_dict['l']=set(j)
                #     to_remove_from_dict.add(("d",j))
                #     to_remove_from_dict.add(("t",j))
                if word.replace(j, i) in ['lin']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen dtl")
                    if set("bpmjqx").issubset(solved_alpha):
                        legal_dict['l'] = set(j)
                    # legal_dict['l']=set(j)
                    to_remove_from_dict.add(("d", j))
                    to_remove_from_dict.add(("t", j))
                if word.replace(j, i) in ['liang']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen dtl")
                    if set("jqx").issubset(solved_alpha):
                        legal_dict['l'] = set(j)
                    # legal_dict['l']=set(j)
                    to_remove_from_dict.add(("d", j))
                    to_remove_from_dict.add(("t", j))
                if word.replace(j, i) in ['den']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen dtl")
                    if set("bpmfgkhzcsr").issubset(solved_alpha):
                        legal_dict['d'] = set(j)
                    # legal_dict['d']=set(j)
                    to_remove_from_dict.add(("l", j))
                    to_remove_from_dict.add(("t", j))
                if word.replace(j, i) in ['tei', 'tiu', 'tin', 'din', 'tia', 'diang', 'tiang', 'lui']:
                    info("remove {} from legal_dict[{}] cos {}/{}".format(
                        j, i, word, word.replace(j, i)), "screen dtl")
                    to_remove_from_dict.add((i, j))
    for i in to_remove_from_dict:
        if i[1] in legal_dict.get(i[0]):
            legal_dict[i[0]].remove(i[1])
        if i[1] in suspects_dict.get(i[0]):
            suspects_dict[i[0]].remove(i[1])
    infoset(zcs_set, "zcs_set:spell_analyse", 12)
    infoset(to_remove_from_dict, "to_remove_from_dict:spell_analyse", 12)
    infodict(legal_dict, "legal_dict:after_screen_zcs", 1, 14)
    infodict(suspects_dict, "suspects_dict:after_screen_zcs", 1, 14)

    match_list = [[i for i in occ_final.keys()], [a for a in occ_final.values()], [
        o for o in exp_final.values()]]
    occ_initial_dict = dict(zip([i for i in occ_final.keys()], [
                            a for a in occ_final.values()]))

    for i, j in exp_initial_dict.items():
        if j:
            for k in j:
                for l, m in occ_initial_dict.items():
                    legal_initial_dict[i] = legal_initial_dict[
                        i].union(occ_initial_dict[l])

    infodict(occ_initial_dict, "occ_initial_dict", 1)
    infodict(legal_initial_dict, "legal_initial_dict", 1)
    infodict(legal_initial_reverse_dict, "legal_initial_reverse_dict", 1)

    lst_tmp = []
    for i in range(len(match_list[2])):
        if match_list[1][i]:
            lst_tmp.append([match_list[1][i], match_list[2][i]])

    infolist(match_list, "match_list", 1, 1)
    infolist(lst_tmp, "lst_tmp", 1, 1)

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
        never_dict[i] = never_dict[i].union(never_dict1[i])
        suspects_dict[i] = suspects_dict[i] - never_dict[i]
        legal_dict[i] = legal_dict[i] - never_dict[i]
        presumably_dict[i] = presumably_dict[i] - never_dict[i]
        probably_dict[i] = probably_dict[i] - never_dict[i]
        possibly_dict[i] = possibly_dict[i] - never_dict[i]

    infoset(zcs_set, "zcs_set:b4_spell_analyse", 1, 14)
    infodict(legal_dict, "legal_dict:b4_spell_analyse", 1, 14)
    infodict(suspects_dict, "suspects_dict:b4_spell_analyse", 1, 14)
    progress = msg
    progress = updater(progress, legal_dict)

    infodict(legal_dict, "legal_dict:spell_analyse", 1, 14)
    infodict(suspects_dict, "suspects_dict:spell_analyse", 1, 14)
    infoset(couples, "couples:spell_analyse", 5)
    infoset(unsolved_symbol, "unsolved_symbol:spell_analyse", 14)
    infoset(remain_alpha, "remain_alpha:spell_analyse", 14)
    infoset(solved_alpha, "solved_alpha:spell_analyse", 14)
    infodict(mapping_dict, "mapping_dict:spell_analyse", 7)
    info(progress, "progress:spell_analyse", 1)

    return progress

def gen_most_used_phrases(max_len=8):
    with open("data/most_used_chinese_word_op.txt", "r", encoding="utf-8") as fp:
        most_used_chinese_word = fp.read().split("\n")
    # 高频词汇的列表，按长度分
    most_used_phrases = [[] for i in range(max_len+1)]
    for x in range(2,max_len+1):
        for i in most_used_chinese_word:
            if i.count(" ") == x-1:
                most_used_phrases[x].append(i)
    return most_used_phrases

def gen_phrase_list(num,msg):
    list_whole = msg.split()
    phrase_list = [[] for i in range(num+1)]
    for i in range(num):
        for x in range(2,i+1):
            for j in range(len(list_whole) - i):
                phrase_list[i].append(" ".join(list_whole[j:j + i]))
    return phrase_list

def find_match_phrase(num,lst):
    most_used_phrases = gen_most_used_phrases(num)
    match_phrase_list = []
    for i in range(num):
        for j in lst[i]:
            if j:
                is_all_lower = True
                for char in j.replace(" ", ""):
                    is_all_lower &= char.islower()
                if not is_all_lower:
                    tmp = re.sub(r'[A-Z]', '[a-z]', j)
                    pat = re.compile('^' + tmp + '$')
                    # info("{}:{}".format(j,pat),"find_match_phrase")
                else:
                    pat = j
                for j in most_used_phrases[i]:
                    g = re.match(pat, j)
                    if g:
                        match_phrase_list.append((j,g.group()))
    return match_phrase_list

def matcher(msg,max_len=8):
    '''
    max_len: int,设定最多统计到几个字的词
    '''
    most_used_phrases = gen_most_used_phrases(max_len)
    list_whole = msg.split()
    # occ_word_list = [[] for i in range(max_len+1)]

    infolist(most_used_phrases,"most_used_phrases:matcher",3)

    # match_phrase_list = []

    # def gen_list(num):
    #     for x in range(2,num+1):
    #         for i in range(len(list_whole) - num):
    #             occ_word_list[num].append(" ".join(list_whole[i:i + num]))

    # def find_match_word(num):
    #     for i in occ_word_list[num]:
    #         if i:
    #             is_all_lower = True
    #             for char in i.replace(" ", ""):
    #                 is_all_lower &= char.islower()
    #             if not is_all_lower:
    #                 tmp = re.sub(r'[A-Z]', '[a-z]', i)
    #                 pat = re.compile('^' + tmp + '$')
    #                 # info("{}:{}".format(i,pat),"find_match_word")
    #             else:
    #                 pat = i
    #             for j in most_used_phrases[num]:
    #                 g = re.match(pat, j)
    #                 if g:
    #                     match_phrase_list.append((i,g.group()))

    def find_same_cap(msg, char):
        counter = pos = 0
        index_pos = []
        while(True):
            pos = msg.find(char , pos)
            if pos > -1:
                index_pos.append(pos)
                counter = counter + 1
                pos = pos + 1
            else:
                break
        return (counter, index_pos)

    def clean_match_phrase_list(match_phrase_list):
        info(len(match_phrase_list),"before")
        removed_from_match_because_cap = set()
        removed_from_match_because_cap1 = set()
        removed_from_match_because_cap2 = set()
        removed_from_match_because_solve1 = set()
        removed_from_match_because_solve2 = set()
        not_removed = set()
        to_removed = []
        match_phrase_list.sort()
        for i in upper_case:
            exclude = False
            for j in match_phrase_list:
                result = find_same_cap(j[0], i)
                companion = set()
                if result[0] > 1:
                    # info("{} : {} / {}".format(i,result[0], result[1]),"")
                    for k in result[1]:
                        companion.add(j[1][k])
                    if not len(companion) == 1:
                        exclude = True
                        # info("{}\t{}\t{}\t{}".format(companion,i,j,exclude), "cap conflict:companion,i,j,exclude")
                        to_removed.append(j)
                        removed_from_match_because_cap.add(j)
                    else:
                        if mapping_dict[i] == j[1][result[1][0]]:
                            # info("{}\t{}\t{}\t{}\t{}\t{}".format(j[0].find(i),i,j,exclude,mapping_dict[i],list(j[0].find(i))[0]), "not conflict:j[0].find(i),i,j,exclude,mapping_dict[i],list(j[0].find(i))[0]")
                            exclude = True
                            removed_from_match_because_cap1.add(j)
                            to_removed.append(j)
                        else:
                            # info("{}\t{}\t{}\t{}\t{}\t{}".format(j[0].find(i),i,j,exclude,mapping_dict[i],list(j[0].find(i))[0]), "solve conflict:j[0].find(i),i,j,exclude,mapping_dict[i],list(j[0].find(i))[0]")
                            exclude = False
                            removed_from_match_because_cap2.add(j)
                            pass
                elif result[0] == 1:
                    if mapping_dict[i]:
                        # info("{}\t{}\t{}\t{}\t{}\t{}".format(j[0].find(i),i,j,mapping_dict[i],j[0].find(i),mapping_dict[i] == j[1][j[0].find(i)]), "not conflict:j[0].find(i),i,j,mapping_dict[i],list(j[0].find(i))[0],mapping_dict[i] == list(j[0].find(i))[0]")
                        if mapping_dict[i] == j[result[1][0]]:
                            exclude = True
                            removed_from_match_because_solve1.add(j)
                        else:
                            exclude = False
                            to_removed.append(j)
                            removed_from_match_because_solve2.add(j)
                    else:
                        exclude = False
                        not_removed.add(j)
            # info("{}\t{}".format(i,exclude))

        for i in to_removed:
            match_phrase_list.remove(i)
        # infolist(removed_from_match_because_cap,"removed_from_match_because_cap",1)
        # infolist(removed_from_match_because_cap1,"removed_from_match_because_cap1",1)
        # infolist(removed_from_match_because_cap2,"removed_from_match_because_cap2",1)
        # infolist(removed_from_match_because_solve1,"removed_from_match_because_solve",1)
        # infolist(removed_from_match_because_solve2,"removed_from_match_because_solve2",1)
        # infolist(not_removed,"not_removed",1)
        # info(len(match_phrase_list),"after")
        # infolist(match_phrase_list,"match_phrase_list:after",1)
        return match_phrase_list

    # for i in range(max_len):
    #     gen_list(i)

    occ_word_list = gen_phrase_list(max_len,msg)
    match_phrase_list = find_match_phrase(max_len,occ_word_list)
    # for i in range(2,max_len):
    #     find_match_word(i)

    infolist(occ_word_list,"occ_word_list:matcher",3)
    info(len(match_phrase_list),"match_phrase_list: before".format(),1)

    match_phrase_list = clean_match_phrase_list(match_phrase_list)

    infoset(set(match_phrase_list),"match_phrase_list: after clean_match_phrase_list".format(),1)

    return match_phrase_list

def estimate_initial(match_phrase_list):
    max_len = min([match_phrase_list.count(i) for i in match_phrase_list])
    for i in match_phrase_list:
        times = [k[1] for k in match_phrase_list].count(i[1]) / max_len
        word_count = len(i[0].split(" "))
        char_count = len(i[0].replace(" ", ""))
        cap_char_count = len(set(re.findall('[A-Z]',i[0])))
        t1= list(set(match_phrase_list))
        t2= [k[0] for k in list(set(match_phrase_list))]
        ambiguity = t1.count(i) / t2.count(i[0])
        perc_by_word = cap_char_count / word_count
        perc_by_char = cap_char_count / char_count
        weight_word_count = round(2**(word_count-4)+0.5, 2)
        weight_char_count = round(2**(char_count/3)+0.5, 2)
        weight_times = round(2**((times-2)*0.6), 2)
        certainty = ((1- perc_by_char)*12 + (1 - perc_by_word)*2+ (word_count-1)*6)*times*(ambiguity+0.5)
        info("char:{}\nword:{}\ntime:{}\nambi:{}\t{}/{}\nscore:{}".format((1- perc_by_char),(1 - perc_by_word),times,ambiguity,t1.count(i),t2.count(i[0]),certainty),"{}".format(i))
        if (cap_char_count / word_count) < 1:
            for j in i[0].split(" "):
                if certainty >= 50 and ambiguity >0.3:
                    for k in j:
                        if k.isupper():
                            idx = i[0].find(k)
                            # info("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i,k,idx,i[0][idx],i[1][idx],certainty,(times / max_len),char_count,str((1 - cap_char_count / char_count)*1.5),((word_count - cap_char_count) / cap_char_count)),"presumably_matched",1)
                            if i[0][idx].strip():
                                if type(presumably_dict[i[1][idx].strip()]) == set:
                                    presumably_dict[i[1][idx].strip()].add(i[0][idx].strip())
                elif certainty >= 25 and times >= 2 and ambiguity >0.3:
                    for k in j:
                        if k.isupper():
                            idx = i[0].find(k)
                            # info("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i,k,idx,i[0][idx],i[1][idx],certainty,(times / max_len),char_count,str((1 - cap_char_count / char_count)*1.5),((word_count - cap_char_count) / cap_char_count)),"probably_matched",1)
                            if i[0][idx].strip():
                                if type(probably_dict[i[1][idx].strip()]) == set:
                                    probably_dict[i[1][idx].strip()].add(i[0][idx].strip())
                elif certainty > 10 and word_count >2 and ambiguity >0.3:
                    for k in j:
                        if k.isupper():
                            idx = i[0].find(k)
                            # info("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i,k,idx,i[0][idx],i[1][idx],certainty,(times / max_len),char_count,str((1 - cap_char_count / char_count)*1.5),((word_count - cap_char_count) / cap_char_count)),"possibly_matched",1)
                            if i[0][idx].strip():
                                if type(possibly_dict[i[1][idx].strip()]) == set:
                                    possibly_dict[i[1][idx].strip()].add(i[0][idx].strip())

def phrase_freq_analyse(msg,max_len=8):
    '''
    '''
    match_phrase_list=matcher(msg,max_len)
    # infolist(match_phrase_list,"match_phrase_list:phrase_freq_analyse")
    estimate_initial(match_phrase_list)
    for i in remain_alpha:
        guess_dict[i]=presumably_dict[i].intersection(legal_dict[i]).intersection(suspects_dict[i])
    progress=msg
    progress = updater(progress,guess_dict)
    progress = updater(progress,legal_dict)
    return progress

def exhaust(lst,msg):
    lst = list(set(lst).intersection(set(remain_alpha)))
    infolist(lst,"actualy letter to exhaust",13)
    list_to_prod = []
    cartesian_prod = []
    tmp_unsolved_symbol=list(set(flat([list(suspects_dict[k]) for k in lst])))
    suspects_reverse_dict = dict(zip(tmp_unsolved_symbol, [[] for i in range(len(tmp_unsolved_symbol))]))
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
        # infoset(tmp_couples,"tmp_couples",5)
        tmp_couple_lst.append(tmp_couples)
    infolist(tmp_couple_lst,"tmp_couple_lst",2)

    tmp_progress_dict = {}
    for i in tmp_couple_lst:
        tmp_progress_dict.update(dict(zip([tuple(i) for i in tmp_couple_lst],['' for i in range(len(tmp_couple_lst))])))
    for i,j in tmp_progress_dict.items():
        info(j,"{}".format(i))
    for i in tmp_couple_lst:
        tmp_progress = msg
        title = []
        for k in i:
            tmp_progress=tmp_progress.replace(k[0],k[1])
            title.append((k[0],k[1]))
        title.sort()
        tmp_progress_dict[tuple(title)] = tmp_progress

    # for i,j in tmp_progress_dict.items():
    #     info(j,"{}".format(i))
    infodict(tmp_progress_dict, 'tmp_progress_dict',1)
    return tmp_progress_dict


    '''
    # list_prod_length = 1
    # list_to_prod = set()
    # cartesian_prod = []
    # tmp_progress_dict = {}
    # new_list = []
    # for i in lst:
    #     list_prod_length *= len(suspects_dict[i])
    # # info("There's {} possibilities\n{}".format(list_prod_length, 'IT IS OK' if not list_prod_length>10000 else 'TO MARCH! I DONT WANT TO TRY!'),"list_prod_length")
    # # if list_prod_length > 10000:
    # #     return
    # # else:
    # for i in lst:
    #     if i not in solved_alpha:
    #         new_list.append(i)
    #         list_to_prod.add(list(suspects_dict[i]))
    #         # info("[{}] add to cartesian_prod".format(','.join(list(suspects_dict[i]))),"add cartesian_prod")

    # infolist(list_to_prod,"list_to_prod",1)
    # for i in itertools.product(*list_to_prod):
    #     cartesian_prod.append(i)
    # cartesian_prod = list(filter(lambda x:x[0] != x[1], cartesian_prod))
    # cartesian_prod_list = [dict(zip(new_list,i)) for i in cartesian_prod]

    # infolist(cartesian_prod,"cartesian_prod",1)
    # infolist(cartesian_prod_list,"cartesian_prod_list",1)
    '''

    # for i in cartesian_prod_list:
    #     tmp_progress_dict.update(dict(zip([tuple(zip(tuple(i),tuple(i.values())))],['' for i in range(len(cartesian_prod_list))])))

    # for i,j in enumerate(cartesian_prod_list):
    #     tmp_progress = msg
    #     title = []
    #     for k,l in j.items():
    #         tmp_progress=tmp_progress.replace(l,k)
    #         title.append((k,l))
    #     # info(tmp_progress,"tmp_progress >> {}: {}".format(str(i),j))
    #     tmp_progress_dict[tuple(title)] = tmp_progress

    # infodict(tmp_progress_dict, 'tmp_progress_dict',1)
    # return tmp_progress_dict

def assess_it(msg,max_len):
    most_used_phrases = gen_most_used_phrases(max_len)

    # def gen_list(num):
    #     for x in range(2,num+1):
    #         for i in range(len(list_whole) - num):
    #             occ_word_list[num].append(" ".join(list_whole[i:i + num]))
    # # tmp_list = gen_word_list(msg)
    # # match_phrase_list = []
    # def find_match_word(num):
    #     for i in tmp_list[num - 2]:
    #         if i:
    #             is_all_lower = True
    #             for char in i.replace(" ", ""):
    #                 is_all_lower &= char.islower()
    #             if not is_all_lower:
    #                 tmp = re.sub(r'[A-Z]', '[a-z]', i)
    #                 pat = re.compile('^' + tmp + '$')
    #             else:
    #                 pat = i
    #             for j in most_used_phrases[num - 2]:
    #                 g = re.search(i, j)
    #                 if g:
    #                     match_phrase_list.append((i, g.group()))

    # list_whole = msg.split(" ")

    # tmp_list = [[] for i in range(max_len)]

    # for k in range(max_len):
    #     gen_list(k)

    # match_phrase_list = []
    # for k in range(max_len):
    #     find_match_word(k)

    tmp_list = gen_phrase_list(max_len,msg)
    match_phrase_list = find_match_phrase(max_len,tmp_list)

    total_matched_times = int(len(match_phrase_list) / max_len)
    matched_phrases = len(set(match_phrase_list))
    match_word_dict = dict(zip(list(set(match_phrase_list)), [0 for k in range(len(set(match_phrase_list)))]))
    for k in match_phrase_list:
        match_word_dict[k] = int(match_phrase_list.count(k) / max_len)

    # infolist(set(match_phrase_list),"match_phrase_list: {} [{}/{} matched]".format(i,len(set(match_phrase_list)),len(match_phrase_list)),1)
    infodict(match_word_dict,"match_word_dict: [{}/{} matched]".format(matched_phrases,total_matched_times),1)

    matched_more_than_one_time = dict(zip([k for k in match_word_dict.keys() if match_word_dict[k] > 1],
        [l for l in match_word_dict.values() if l > 1]))

    # infodict(matched_more_than_one_time,"matched_more_than_one_time: {} [{}/{} matched]".format(i,matched_phrases,total_matched_times),1)

    the_items = [k[1] for k in match_word_dict.keys()]
    lengths = [len(k[0].split(" ")) for k in match_word_dict.keys()]
    times = [match_word_dict[k] for k in match_word_dict.keys()]
    max_length = max(lengths)
    max_times= max(times)
    max_length_item = sorted(match_word_dict.items(),key = lambda x:len(x[0]),reverse=True)[0]
    max_times_item = sorted(match_word_dict.items(),key = lambda x:x[1],reverse=True)[0]
    weight_length = [round(2**(k-4)+0.5, 2) for k in range(1, max_len+1)]
    weight_times = [round(2**((k-2)*0.6), 2) for k in range(1, max_len+1)]
    score_list = [round(k**weight_length[k]+l*weight_times[l], 2) for k,l in zip(lengths, times)]
    score = sum(score_list)

    case_score_dict = dict(zip(the_items,score_list))

    info("{} : {}\t\t{} : {}".format(max_length_item, max_length, max_times_item, max_times), "max",1)
    infolist(sorted(match_word_dict.items(),key = lambda x:len(x[0]),reverse=True), "by_length",1)
    infolist(sorted(match_word_dict.items(),key = lambda x:x[1],reverse=True), "by_times",1)
    infolist(lengths, "lengths",10)
    infolist(times, "times",10)
    infolist([str(k) for k in weight_length], "weight_length:{}".format(str(len(weight_length))), 5)
    infolist([str(k) for k in weight_times], "weight_times:{}".format(str(len(weight_times))), 5)
    # infolist(score_list, "score_list:{}".format(str(len(score_list))), 3)
    # infolist(the_items, "the_items:{}".format(str(len(the_items))), 3)
    infodict(case_score_dict, "case_score_dict:{}".format(str(len(case_score_dict))), 2)
    # info("{} : {}/{}".format(i, score, score1), "score")
    return score

def compare_it(progress_dic,max_len):
    assess_dict = dict(zip(progress_dic.keys(),[0 for i in range(len(progress_dic))]))
    for i,j in progress_dic.items():
        var_holder = ["{} -> {}".format(p[1],p[0]) for p in i]
        info("[replace {}]:\n{}".format(', '.join(var_holder),j), "compare_it")

        score = assess_it(j,max_len)
        assess_dict[i] = score
    better = sorted(assess_dict, key=lambda x: x, reverse=False)[0]
    infodict(assess_dict, "assess_dict:{}".format(str(len(assess_dict))), 1)
    info("{} : {} is better\n-----\n{}".format(better,assess_dict[better],progress_dic[better]), "better")
    # info("{}\n-----\n{}".format(progress_dic[better],), "tmp_progress:better")
    for i in better:
        couples.add((i[0], i[1]))

def brute_crack(msg,max_len=5):
    progress = msg
    progress = updater(progress, legal_dict)
    progress = updater(progress, suspects_dict)
    progress = updater(progress, legal_dict)
    tmp_progress_dict_zcs = exhaust(list("zcs"), progress)
    compare_it(tmp_progress_dict_zcs,max_len)
    progress = updater(progress, legal_dict)
    if remain_alpha:
        tmp_progress_dict = exhaust(remain_alpha, progress)
        compare_it(tmp_progress_dict,max_len)
        progress = updater(progress, legal_dict)
    # progress = updater(progress, suspects_dict)
    # if remain_alpha:
    #     tmp_progress_dict = exhaust(remain_alpha, progress)
    #     compare_it(tmp_progress_dict,max_len)
    #     progress = updater(progress, legal_dict)
    #     progress = updater(progress, suspects_dict)
    return progress

# fname="留言2_op_已加密_转写"
# fname="来自白犀的问候_op_已加密_转写"
# fname="因为门德尔松_op_已加密_转写"
# fname="随想_op_已加密_转写"
fname = "留言1-1_op_已加密_转写"
with open("src/" + fname + ".txt", "r", encoding="utf-8") as fp:
    article = fp.read()
    article_without_space = article.replace(" ", "")
    total_chars = len(article_without_space)
    total_words = len(article.split())
    occ_symbos = list(set(article_without_space))
    total_symbos = len(occ_symbos)
    info("chars:{}, words:{}, symbos:{}".format(
        total_chars, total_words, total_symbos))
    infolist(occ_symbos, "occ_symbos", 14)
    lower_case = [chr(i) for i in range(97, 123)]
    upper_case = [chr(i) for i in range(65, 91)]
    # 待替换的字母对 (大写字母, 小写字母)
    couples = set()
    # 明码密码对照表
    mapping_dict = dict(zip(upper_case[:total_symbos], [
                        '' for i in range(total_symbos)]))
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
    # 存放其它字典运算结果的临时名单
    guess_dict = dict(zip(lower_case, [set() for i in range(26)]))

progress, legal_pinyin_list = letter_pos_analyse(fname)
progress = letter_freq_analyse(progress)
progress = spell_analyse(progress)
progress = spell_analyse(progress)
progress = updater(progress, legal_dict)
progress = phrase_freq_analyse(progress,max_len=5)
# progress = updater(progress, {k: suspects_dict[k] for k in suspects_dict.keys() if k in remain_alpha})
progress = brute_crack(progress,max_len=5)
# progress = updater(progress, legal_dict)
# progress = updater(progress, suspects_dict)
# progress="Au Ai Dhuo Ai lv Ie Kai Mi guang hua Ie Dhi Ping lan gao Ia Ie Qao Pia Dhu Qi hong Ie Dang Qhen Re Au Ai Dhuo Sing Khan Qai Dhu Re li Khang Rin Tei Uang Ie huang Teng Tu Qai Kai hua Dhang Ming Pie Ie Piao Vian Qi hu ran Kong Kao Pian Qhi Kuan Xiang Run Xiao li Mu le o Ri Dhang Qhi Dhi gen ni Sen Yai ge Zan Xiao gong Xi ni Sen Vong guo le Ii Dan guan Xian Qai ni Sen Xiao Qu Ri Ping Qheng Sing le Qi Pi Ie Dhi li Sing nian Iong Vian Piang Rou Ri Khang Ra Qhou Dhao nian er Vong Si Sa Ia Piang Dai Zo Qhen Kheng Ii Xi Zang ni Sen Pie Dhi Re neng gou Kan Pia"


infodict({k: possibly_dict[k] for k in possibly_dict.keys() if k in remain_alpha}, "possibly_dict:now", 1, 14)
infodict({k: probably_dict[k] for k in probably_dict.keys() if k in remain_alpha}, "probably_dict:now", 1, 14)
infodict({k: presumably_dict[k] for k in presumably_dict.keys() if k in remain_alpha}, "presumably_dict:now", 1, 14)
infodict({k: guess_dict[k] for k in guess_dict.keys() if k in remain_alpha}, "guess_dict:now", 1, 14)
infodict({k: legal_dict[k] for k in legal_dict.keys()
          if k in remain_alpha}, "legal_dict:now", 1, 14)
infodict({k: suspects_dict[k] for k in suspects_dict.keys(
) if k in remain_alpha}, "suspects_dict:now", 1, 14)
infoset(couples, "couples:now", 5)
infoset(unsolved_symbol, "unsolved_symbol:now", 14)
infoset(remain_alpha, "remain_alpha:now", 14)
infoset(solved_alpha, "solved_alpha:now", 14)
infodict(mapping_dict, "mapping_dict:now", 7)
info(progress, "progress:now", 1)
