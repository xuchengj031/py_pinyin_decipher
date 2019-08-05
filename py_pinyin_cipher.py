import re

from py_formatted_log import *

lower_case = [chr(i) for i in range(97, 123)]
upper_case = [chr(i) for i in range(65, 91)]
# 待替换的字母对 (大写字母, 小写字母)
couples = set()
# 明码密码对照表
mapping_dict = dict(zip(upper_case, ['' for i in range(26)]))
# 未破译的密文符号 (大写)
unsolved_symbol = [chr(i) for i in range(65, 91)]
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

guess_dict = dict(zip(lower_case, [set() for i in range(26)]))


def gen_legal_pinyin_list(msg):
    lst = msg.split() if type(msg) == str else msg
    word_max_len = max([len(i) for i in lst])
    legal_pinyin_list = [set() for i in range(word_max_len + 1)]
    for word in lst:
        for x in range(1, len(lst)):
            if len(word) == x:
                legal_pinyin_list[x].add(word)
    return legal_pinyin_list


def stat_pos(msg, char_list):
    legal_pos_dict = dict(
        zip(char_list, [set() for x in range(len(char_list))]))
    legal_pinyin_list = gen_legal_pinyin_list(msg)
    for letter in char_list:
        for i in legal_pinyin_list:
            for j in i:
                if j.find(letter) != -1:
                    legal_pos_dict[letter].add((len(j), j.find(letter) + 1))
    return (legal_pos_dict, legal_pinyin_list)


def verifier(c_char, p_char, msg):
    '''
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

    info("{} -> {}: {} {} {} {} {} {}".format(c_char, p_char,cond1,cond2,cond3,cond4,cond5,cond6),"verifier")
    return all([cond1, cond2, cond3, cond4, cond5, cond6])


def committer(c_char, p_char, msg):
    '''替换 (符号, 字母) 对
    '''
    c_char = list(c_char)[0] if type(c_char) == set else str(c_char)
    progress = msg
    progress = progress.replace(c_char, p_char)
    info(progress[:300],'committer {} - {}'.format(c_char, p_char))
    unsolved_symbol.remove(c_char)
    remain_alpha.remove(p_char)
    solved_alpha.append(p_char)
    mapping_dict[c_char] = p_char
    return progress


def updater(msg, dic):
    '''
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
                if j in list("zcs"):
                    set_zcs = set_zcs - set(i)
        for i in to_remove:
            for j, k in dict_to_update.items():
                if type(k) == set:
                    if j != i[0] and i[1].issubset(k):
                        dict_to_update[j] = dict_to_update[j] - i[1]
        # 所有嫌疑字典里都排除掉never字典里的名单
        for i in remain_alpha:
            for j in [legal_dict]:
            # for j in [legal_dict, suspects_dict, presumably_dict, probably_dict, possibly_dict, guess_dict]:
                if j:
                    j[i] = j[i] - never_dict[i]

        # 返回该字典里剩余具有唯一值的键的个数
        engaged = [k for k in dict_to_update.values() if len(k) == 1]
        info("engaged unique: {}\t{}".format(engaged, len(engaged)),'update_dict')

        return len([k for k in dict_to_update.values() if len(k) == 1])

    def check_unique_value(dict_to_check):
        all_solved = set([i for i in dict_to_check.keys() if len(
            dict_to_check[i]) == 1]).issubset(set(solved_alpha))
        info(all_solved,'check_unique_value')
        return all_solved

    def update_couples(msg):
        progress = msg

        for i in list(couples):
            if verifier(i[0], i[1], msg):
                progress = committer(i[0], i[1], progress)
                info((i[0], i[1]),'update_couples')

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


def main(fname):

    with open('data/all_legal_pinyin.txt', 'r', encoding="utf-8") as fp:
        all_legal_pinyin = fp.read()
    # infolist(all_legal_pinyin.split(), "all_legal_pinyin", 7)

    # 统计正确拼音中每个字母的位置
    legal_pos_dict, legal_pinyin_list = stat_pos(all_legal_pinyin, lower_case)

    # fname = "留言1-1_op_已加密_转写"
    with open("src/" + fname + ".txt", "r", encoding="utf-8") as fp:
        article = fp.read()
        total_chars = len(article.replace(" ", ""))
        total_words = len(article.split())

    # 统密文中每个字母的位置
    occ_pos_dict, occ_pinyin_list = stat_pos(article, upper_case)

    # 如果密文中符号的位置的集合，正好是某字母正确位置的子集，那它可能是这个字母；
    # 反之如果不在该字母位置集合的范围内，它就绝对不会是这个字母
    for i, j in occ_pos_dict.items():
        for k, l in legal_pos_dict.items():
            if j.issubset(l):
                legal_dict[k] = legal_dict[k].union(i)
                legal_dict_copy[k] = legal_dict[k].union(i)
            else:
                never_dict[k] = never_dict[k].union(i)

    # infodict(legal_dict, "legal_dict_0", 1, 13)
    # infodict(never_dict, "never_dict_0", 1, 13)

    has_single_pos = [i for i in legal_pos_dict.keys() if max(
        [j[1] for j in legal_pos_dict[i]]) < 2]
    has_multi_pos = list(set(lower_case) - set(has_single_pos))
    has_single_pos_set = set()
    has_multi_pos_set = set()
    for i in occ_pos_dict.keys():
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

    # infolist(legal_pinyin_list, "legal_pinyin_list", 1, 7)
    # infolist(occ_pinyin_list, "occ_pinyin_list", 1, 7)
    # infodict(legal_pos_dict, "legal_pos_dict", 1, 6)
    # infodict(occ_pos_dict, "occ_pos_dict", 1, 6)
    infosimple(has_single_pos,"has_single_pos",20)
    infosimple(has_single_pos_set,"has_single_pos_set",20)
    infosimple(has_multi_pos,"has_multi_pos",20)
    infosimple(has_multi_pos_set,"has_multi_pos_set",20)
    # infodict(never_dict, "never_dict", 1, 13)
    infodict(legal_dict, "legal_dict", 1, 13)

    progress = updater(article, legal_dict)

    infoset(couples, "couples", 5)
    infoset(unsolved_symbol, "unsolved_symbol", 13)
    infoset(remain_alpha, "remain_alpha", 13)
    infoset(solved_alpha, "solved_alpha", 13)
    infodict(mapping_dict, "mapping_dict", 6)
    info(progress, "progress", 1)

main("留言1-1_op_已加密_转写")
