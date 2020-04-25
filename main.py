# %%
from pathlib import Path
from glob import glob
import genanki
import math
import re
from decimal import Decimal
import pandas as pd
import json
import dominate
from dominate import dom_tag
from dominate.tags import div, span, details, summary
from typing import *

df = pd.DataFrame(columns=['Hanzi', 'Order', 'Traditional', 'Frequency',
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

# %% dictionary

dictionary_type = Dict[str, Tuple[str, List[Tuple[str, List[str]]]]]

def read_dictionary(path: str) -> dictionary_type:
    dictionary: dictionary_type = {}

    with open(path) as dictionary_file:
        for line in dictionary_file.readlines():
            if line[0] == '#':
                # skip comments
                continue
            m = re.match('(.*) (.*) \\[(.*)\\] /(.*)/', line)
            if not m:
                raise ValueError(f"Bad format on line {line}")

            [traditional, simplified, pinyin, joined_meanings] = m.groups()

            dictionary.setdefault(simplified, (traditional, []))
            # a bit hacky
            dictionary[simplified] = (
                traditional,
                dictionary[simplified][1] + [
                    (
                            pinyin,
                            [meaning for meaning in joined_meanings.split('/')]
                    )
                ]
            )
            

    return dictionary

dictionary = read_dictionary('data/cedict_ts.u8')

# %% entries
def entries(df: pd.DataFrame, dictionary: dictionary_type):

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
 
    # fill entries_dict from file
   
    for simplified in df.index:
        if simplified in dictionary:
            entries_html = div(cls='entries')
            with entries_html:
                for pinyin, meanings in dictionary[simplified][1]:
                    entry_html = div(cls='entry')
                    with entry_html:
                        div(decode_pinyin(pinyin), cls=f'pinyin tone-{pinyin[-1]}')
                        div(', '.join(meanings), cls='definition')

            df.at[simplified, 'Entry'] = entries_html

entries(df, dictionary)

# %% traditional
def traditional(df: pd.DataFrame, dictionary: dictionary_type):
    for simplified in df.index:
        if simplified in dictionary:
            traditional, _ = dictionary[simplified]
            df.at[simplified, 'Traditional'] = traditional

traditional(df, dictionary)

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

with open('data/outlier_decomp.json') as _d2:
    d2 = json.load(_d2)
for _k in df.index:
    components_dict[_k] = (d2[_k] if _k in d2 else [])


def decompose(h):
    """
    Decomposes words into its characters and characters into components recursively.
    """
    def decompose_component(c):
        components = []
        if c in components_dict:
            components = components_dict[c]
        return (c, [decompose_component(subcomponent) for subcomponent in components])


    if len(h) > 1:
        # word
        return (h, [decompose_component(hanzi) for hanzi in h])
    else:
        return decompose_component(h)


def process_components(cs) -> str:
    def _process_components(cs, depth):
        component, subcomponents = cs
        with div() as entries:
            if component in df.index:
                if depth == 0 or not df.at[component, 'Entry']:
                    # do not collapse first component or if no entries
                    with div() as entry:
                        entry += span(component, cls='hanzi entry-title')
                        entry += df.at[component, 'Entry']
                        for subcomponent in subcomponents:
                            entry += _process_components(subcomponent, depth + 1)
                else:
                    with details() as ds:
                        with summary():
                            span(component, cls='hanzi entry-title')
                        ds += df.at[component, 'Entry']
                        for subcomponent in subcomponents:
                            ds += _process_components(subcomponent, depth + 1)
        return entries

    return _process_components(cs, 0).render(pretty=False)

# %% cleanup

df.fillna('', inplace=True)

# %%

ultimate_deck = genanki.Deck(1251384833, name='Chinese::Ultimate')

ultimate_package = genanki.Package(ultimate_deck)
ultimate_package.media_files = glob("audio/*.mp3")


ultimate_model = genanki.Model(
    1693520247,
    name='Ultimate',
    fields=[
        {'name': 'Sort Field'},
        {'name': 'Simplified'},
        {'name': 'Traditional'},
        {'name': 'Frequency'},
        {'name': 'Strokes'},
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
            str(row['Traditional']),
            str(row['Frequency']),
            str(row['Strokes']),
            process_components(decompose(row.name)),
            str(row['Comment']),
            f"[sound:ultimate_{row['Order']}.mp3]",
        ])


_count = 0 # debug
for i, row in df.iterrows():
    # if _count > 100:
    #     break

    # do not add if entry is empty
    # if not row['Entry']:
    #   continue
    
    n = note(row)
    ultimate_deck.add_note(note(row))

    # print(n.fields)
    _count += 1

ultimate_package.write_to_file('ultimate.apkg')


# %%


# %%
