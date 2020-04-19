# %%
from pathlib import Path
from glob import glob
import genanki
import math
import re
from decimal import Decimal
import pandas as pd
import json

df = pd.DataFrame(columns=['Hanzi', 'Order', 'Frequency',
                           'Strokes', 'Components', 'Entry', 'Comment'])
with open('data/loach_word_order.json') as lo:
    loach_order = json.load(lo)
df['Hanzi'] = [h for h in loach_order]
df['Order'] = df.index + 1

df = df.set_index('Hanzi')

# %% Frequency

with open('data/word_freq.json') as _f_f:
    _frequencies = json.load(_f_f)
for _h in df.index:
    df.at[_h, 'Frequency'] = "{:.2E}".format(
        Decimal(_frequencies[_h])) if _h in _frequencies else ''

# %% Strokes
with open('data/char_strokes.json') as _s_f:
    _s = json.load(_s_f)
for _k in df.index:
    df.at[_k, 'Strokes'] = _s[_k] if _k in _s else ''

# %% entries


def decode_pinyin(s):
    PinyinToneMark = {
        0: "aoeiuv\u00fc",
        1: "\u0101\u014d\u0113\u012b\u016b\u01d6\u01d6",
        2: "\u00e1\u00f3\u00e9\u00ed\u00fa\u01d8\u01d8",
        3: "\u01ce\u01d2\u011b\u01d0\u01d4\u01da\u01da",
        4: "\u00e0\u00f2\u00e8\u00ec\u00f9\u01dc\u01dc",
    }

    s = s.lower()
    r = ""
    t = ""
    for c in s:
        if c >= 'a' and c <= 'z':
            t += c
        elif c == ':':
            assert t[-1] == 'u'
            t = t[:-1] + "\u00fc"
        else:
            if c >= '0' and c <= '5':
                tone = int(c) % 5
                if tone != 0:
                    m = re.search("[aoeiuv\u00fc]+", t)
                    if m is None:
                        t += c
                    elif len(m.group(0)) == 1:
                        t = t[:m.start(
                            0)] + PinyinToneMark[tone][PinyinToneMark[0].index(m.group(0))] + t[m.end(0):]
                    else:
                        if 'a' in t:
                            t = t.replace("a", PinyinToneMark[tone][0])
                        elif 'o' in t:
                            t = t.replace("o", PinyinToneMark[tone][1])
                        elif 'e' in t:
                            t = t.replace("e", PinyinToneMark[tone][2])
                        elif t.endswith("ui"):
                            t = t.replace("i", PinyinToneMark[tone][3])
                        elif t.endswith("iu"):
                            t = t.replace("u", PinyinToneMark[tone][4])
                        else:
                            t += "!"
            r += t
            t = ""
    r += t
    return r


entries_dict = {}

with open('data/cedict_ts.u8') as dictionary:
    for line in dictionary.readlines():
        if line[0] == '#':
            continue
        m = re.match('(.*) (.*) \\[(.*)\\] /(.*)/', line)
        [_, simplified, pinyin, joined_meanings] = m.groups()

        if simplified not in entries_dict:
            entries_dict[simplified] = []

        entries_dict[simplified] += [(pinyin,
                                      [meaning for meaning in joined_meanings.split('/')])]

for k in df.index:
    if k in entries_dict:
        df.at[k, 'Entry'] = '\n'.join([f"<div class=\"entry\">{entry}</div>"
                                       for entry in [f"<div class=\"pinyin tone-{pinyin[-1]}\">{decode_pinyin(pinyin)}</div> <div class=\"definition\">{', '.join(meanings)}</div>"
                                                     for (pinyin, meanings) in entries_dict[k]]])

df.fillna('', inplace=True)

# %% radicals
with open('data/radicals.json') as radicals_file:
    radicals = json.load(radicals_file)

for radical in radicals:
    entry = f"{radical['pronunciation']}: ({radical['simplified']}) {', '.join(radical['meanings'])}"
    # original radical
    if radical['simplified'] in df:
        df.at[radical['simplified'], 'Entry'] = entry
        df.at[radical['simplified'], 'Comment'] = radical['comment']
    if radical['variants']:
        # radical variants
        for variant in radical['variants'].split('，'):
            if variant in df:
                df.at[variant, 'Entry'] = entry
                df.at[variant, 'Comment'] = radical['comment']

# %% Decomposition


components_dict = {}

with open('data/outlier_decomp.json') as d2:
    d2 = json.load(d2)
for k in df.index:
    components_dict[k] = (d2[k] if k in d2 else [])


def decompose(h):
    """
    Decomposes words into its characters and characters into components recursively.
    """
    tree = []
    if len(h) > 1:
        for hanzi in h:
            tree.append(hanzi)
            if (ds := decompose(hanzi)):
                tree.append(ds)
    else:
        if h in components_dict:
            for component in components_dict[h]:
                tree.append(component)
                if (ds := decompose(component)):
                    tree.append(ds)
    return tree


def mapover(xs, f):
    res = []
    for x in xs:
        if type(x) == list:
            res.append(mapover(x, f))
        else:
            res.append(f(x))
    return res


def fentry(e):
    component_entry = ""
    try:
        component_entry = df.at[e, 'Entry']
    except:
        component_entry = ''
    return f"<div class=\"component-hanzi\">{e}</div>\n<div class=\"component-entry\">{component_entry}</div>\n"


def fold_component(xs):
    res = ""
    for x in xs:
        res += "<div class=\"component\">\n"
        if type(x) == list:
            res += fold_component(x)
        else:
            res += x
        res += "</div>\n"
    return res


# %%

ultimate_deck = genanki.Deck(1251384833, name='Chinese::Ultimate')

ultimate_package = genanki.Package(ultimate_deck)
ultimate_package.media_files = glob("audio/*.mp3")


ultimate_model = genanki.Model(
    1693520247,
    name='Ultimate',
    fields=[
        {'name': 'Sort Field'},
        {'name': 'Hanzi'},
        {'name': 'Frequency'},
        {'name': 'Strokes'},
        {'name': 'Components'},
        {'name': 'Entry'},
        {'name': 'Comment'},
        {'name': 'Audio'}
    ],
    templates=[
        {
            'name': 'Ultimate-Template',
            'qfmt': Path("question.html").read_text(),
            'afmt': Path("answer.html").read_text(),
        },
    ],
    css = Path("style.css").read_text()
)


class UltimateNote(genanki.Note):
    @property
    def guid(self):
        return genanki.guid_for(self.fields[1])  # only hash the hanzi field


def note(row):
    return UltimateNote(
        model=ultimate_model,
        fields=[
            str(row['Order']),
            str(row.name),
            str(row['Frequency']),
            str(row['Strokes']),
            str(fold_component(mapover(decompose(row.name), fentry))),
            row['Entry'],
            str(row['Comment']),
            f"[sound:ultimate_{row['Order']}.mp3]",
        ])


for _, row in df.iterrows():
    # do not add if entry is empty
    if row['Entry']:
        n = note(row)
        ultimate_deck.add_note(note(row))

ultimate_package.write_to_file('ultimate.apkg')


# %%


# %%
