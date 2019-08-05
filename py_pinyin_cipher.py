from py_formatted_log import *


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


def main():
    lower_case = [chr(i) for i in range(97, 123)]
    with open('data/all_legal_pinyin.txt', 'r', encoding="utf-8") as fp:
        all_legal_pinyin = fp.read()
    # infolist(all_legal_pinyin.split(), "all_legal_pinyin", 7)

    # 统计正确拼音中每个字母的位置
    legal_pos_dict, legal_pinyin_list = stat_pos(all_legal_pinyin, lower_case)

    upper_case = [chr(i) for i in range(65, 91)]
    fname = "留言1-1_op_已加密_转写"
    with open("src/" + fname + ".txt", "r", encoding="utf-8") as fp:
        article = fp.read()
        total_chars = len(article.replace(" ", ""))

    # 统密文中每个字母的位置
    occ_pos_dict, occ_pinyin_list = stat_pos(article, upper_case)

    # 生成反向的位置表，即以位置为键
    legal_pos_list, legal_pos_reverse_dict = gen_reverse_dict(legal_pos_dict)
    occ_pos_list, occ_pos_reverse_dict = gen_reverse_dict(occ_pos_dict)

    # 由字母位置得出，是密文的最大嫌疑范围，逐步缩窄
    legal_dict = dict(zip(lower_case, [set() for i in range(26)]))
    # 留个备份，以免误操作
    legal_dict_copy = dict(zip(lower_case, [set() for i in range(26)]))
    # 绝对排除嫌疑
    never_dict = dict(zip(lower_case, [set() for i in range(26)]))

    # 如果密文中符号的位置的集合，正好是某字母正确位置的子集，那它可能是这个字母；
    # 反之如果不在该字母位置集合的范围内，它就绝对不会是这个字母
    for i, j in occ_pos_dict.items():
        for k, l in legal_pos_dict.items():
            if j.issubset(l):
                legal_dict[k] = legal_dict[k].union(i)
                legal_dict_copy[k] = legal_dict[k].union(i)
            else:
                never_dict[k] = never_dict[k].union(i)

    infodict(legal_dict, "legal_dict", 1, 13)
    infodict(never_dict, "never_dict", 1, 13)

    has_single_pos = [i for i in legal_pos_dict.keys() if max([j[1] for j in legal_pos_dict[i]]) < 2]
    has_multi_pos = list(set(lower_case) - set(has_single_pos))
    has_single_pos_set = set()
    has_multi_pos_set = set()
    for i in occ_pos_dict.keys():
        if max([j[1] for j in occ_pos_dict[i]]) < 2:
            has_single_pos_set.add(i)
        else:
            has_multi_pos_set.add(i)

    for i in has_multi_pos:
        legal_dict[i] = legal_dict[i] - has_single_pos_set
    for i in has_single_pos:
        legal_dict[i] = legal_dict[i] - has_multi_pos_set

    lucky_dict = dict(zip(lower_case, [set() for i in range(26)]))
    reverse_lucky_dict = dict(zip(upper_case, [set() for i in range(26)]))
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

    infolist(legal_pinyin_list, "legal_pinyin_list", 1, 7)
    infolist(occ_pinyin_list, "occ_pinyin_list", 1, 7)
    infodict(legal_pos_dict, "legal_pos_dict", 1, 6)
    infodict(occ_pos_dict, "occ_pos_dict", 1, 6)
    infosimple(has_single_pos,"has_single_pos",20)
    infosimple(has_single_pos_set,"has_single_pos_set",20)
    infosimple(has_multi_pos,"has_multi_pos",20)
    infosimple(has_multi_pos_set,"has_multi_pos_set",20)

    infolist(legal_pos_list, "legal_pos_list", 7)
    infolist(occ_pos_list, "occ_pos_list", 7)
    infodict(legal_pos_reverse_dict, "legal_pos_reverse_dict", 1, 13)
    infodict(occ_pos_reverse_dict, "occ_pos_reverse_dict", 1)
    # infodict(occ_pos_reverse_dict, "occ_pos_reverse_dict", 1, 13)
    infodict(legal_dict, "legal_dict", 1, 13)
    infodict(never_dict, "never_dict", 1, 13)
    infodict(lucky_dict, "lucky_dict", 1, 13)
    infodict(reverse_lucky_dict, "reverse_lucky_dict", 1, 13)
main()
