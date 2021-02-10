import json
import conlleval


def get_res(eval_file, out, decode, mode="normal"):
    if mode == "normal":
        get_result(eval_file, out, decode)
    elif mode == "X":
        get_x_result(eval_file, out, decode)
    elif mode == "focus_weight":
        get_weight_focus_result(eval_file, out, decode)
    elif mode == "weight":
        get_weight_result(eval_file, out, decode)


def is_consistent_label_word(logits):
    if len(logits) == 1:
        return True
    pre_logist = None
    for i, logit in enumerate(logits):
        if not pre_logist:
            pre_logist = logit
        elif pre_logist != logit:
            return False
        else:
            pre_logist = logit
    return True


def get_result(eval_file, out, decode):
    with open(eval_file, 'r') as f:
        lines = json.load(f)
    inconsistent_label_in_one_word_num = 0
    all_logit = []
    all_label = []
    for l in lines:
        logit = []
        label = []
        one_word_logits = []
        one_word_piece = []
        one_word_label = []
        length = len(l['is_start_label'])
        for i, m in enumerate(l['is_start_label']):
            if i == 0 or l['mask'][i] == 0 or l['piece_list'][i] == "[SEP]":
                continue
            one_word_logits.append(l["logits"][i])
            one_word_piece.append(l["piece_list"][i])
            one_word_label.append(l["labels"][i])
            if i + 1 < length and l['is_start_label'][i + 1] == 0:
                continue
            else:
                most_possible_label = get_possible_logit(one_word_logits)
                logit.append(most_possible_label)
                label.append(one_word_label[0])
                one_word_logits = []
                one_word_piece = []
                one_word_label = []
        all_logit.extend(logit)
        all_label.extend(label)
    get_output_file(all_logit, all_label, decode, out)


def get_possible_logit(one_word_logits):
    most_possible_label = None
    logits_kind= []
    logits_num = []
    most_possible_label = None
    for l in one_word_logits:
        if l not in logits_kind:
            logits_kind.append(l)
            logits_num.append(1)
        else:
            i = logits_kind.index(l)
            logits_num[i] += 1
    most_possible_num = 0
    for i,num in enumerate(logits_num):
        if i == 0:
            most_possible_num = num
            most_possible_label = logits_kind[i]
        elif num > most_possible_num:
            most_possible_num = num
            most_possible_label = logits_kind[i]
    return most_possible_label

def get_x_result(eval_file, out, decode):
    with open(eval_file, 'r') as f:
        lines = json.load(f)
    all_logit = []
    all_label = []
    for l in lines:
        logits = []
        labels = []
        for i in range(1, sum(l['mask']) - 1):
            if l['is_start_label'][i] == 1:
                logits.append(l['logits'][i])
                labels.append(l['label_x'][i])
        all_logit.extend(logits)
        all_label.extend(labels)
    get_output_file(all_logit, all_label, decode, out)


def get_weight_focus_result(eval_file, out, decode):
    with open(eval_file, 'r') as f:
        lines = json.load(f)
    all_logit = []
    all_label = []
    for l in lines:
        logits = []
        labels = []
        for i in range(1, sum(l['mask']) - 1):
            if decode[l['weight_focus_label'][i]] != "X":
                logits.append(l['logits'][i])
                labels.append(l['weight_focus_label'][i])
        all_label.extend(labels)
        all_logit.extend(logits)
    get_output_file(all_logit, all_label, decode, out)


def get_weight_result(eval_file, out, decode):
    with open(eval_file, 'r') as f:
        lines = json.load(f)
    all_logit = []
    all_label = []
    for l in lines:
        words_num = sum(l['is_start_label'])
        all_logit.extend(l["logits"][1:words_num - 1])
        all_label.extend(l["labels"][1:words_num - 1])
    get_output_file(all_logit, all_label, decode, out)


def get_result_noavarge(eval_file, out, decode):
    with open(eval_file, 'r') as f:
        lines = json.load(f)
    all_logit = []
    all_label = []
    for l in lines:
        logit = []
        label = []
        for i, m in enumerate(l['is_start_label']):
            if i == 0 or l['mask'][i] == 0 or l['piece_list'][i] == "[SEP]":
                continue
            if m:
                logit.append(l['logits'][i])
                label.append(l['labels'][i])
        # print(l['piece_list'])
        # print(label)
        # break
        all_logit.extend(logit)
        all_label.extend(label)
    get_output_file(all_logit, all_label, decode, out)


def get_output_file(all_logit, all_label, decode, out):
    decode.pop(len(decode) - 1)
    assert len(all_logit) == len(all_label)
    evalseq = []
    for i in range(len(all_logit)):
        evalseq.append("{} {} {}".format(i, decode[int(all_label[i])] if int(all_label[i]) in decode.keys() else "O",
                                         decode[int(all_logit[i])] if int(all_logit[i]) in decode.keys() else "O", ))

    count = conlleval.evaluate(evalseq)
    conlleval.report(count, out)
